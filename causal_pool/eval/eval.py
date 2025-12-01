#!/usr/bin/env python3
"""
Async evaluation script for causal pool dataset.

This script evaluates a model on a dataset with support for:
- Async concurrent evaluation
- Retry logic with tenacity
- Multiple samples per question
- Per-question and per-option accuracy metrics
"""

# Set thread limits before importing any libraries that might spawn threads
# This prevents excessive thread creation on login nodes
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import asyncio
import itertools
import json
import random
import signal
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import jsonlines
from openai import AsyncOpenAI
from openai import BadRequestError, APIError, APIConnectionError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    retry_if_not_exception_type,
    wait_none,
    RetryCallState,
)
from tqdm import tqdm

from causal_pool.eval.eval_utils import (
    InvalidPredictionError,
    get_model_hyperparameters,
    build_prompt,
    get_available_videos,
)
from causal_pool.metrics import (
    calculate_per_question_accuracy,
    calculate_per_option_accuracy,
)
from causal_pool.data.dataset_utils import gather_test_dataset
from causal_pool.utils import normalize_model_name
from causal_pool.prompt_utils import index_to_letter


# Create retry condition by combining existing helpers
# Retry on InvalidPredictionError OR API errors OR (everything except BadRequestError and ValueError)
# Note: InvalidPredictionError is explicitly included even though it's a ValueError subclass
_retry_condition = (
    retry_if_exception_type((InvalidPredictionError, APIError, APIConnectionError, APITimeoutError))
    | retry_if_not_exception_type((BadRequestError, ValueError))
)


def _num_correct_options_from_entry(entry: Dict[str, Any]) -> int:
    """
    Infer the number of correct options from an entry's ground_truth field.

    Handles ground_truth represented as:
    - list/set/tuple of indices or letters
    - single int index
    - string of letters
    """
    ground_truth = entry.get("ground_truth")

    if isinstance(ground_truth, (list, set, tuple)):
        return len(ground_truth)
    if isinstance(ground_truth, int):
        return 1
    if isinstance(ground_truth, str):
        return len(ground_truth)

    try:
        return len(ground_truth)
    except TypeError:
        # Fallback for unexpected types
        return 1


def validate_and_get_metrics(entry: Dict[str, Any], pred: str) -> Tuple[int, int, str]:
    """
    Validate prediction format and calculate metrics using metrics.py functions.
    
    Args:
        entry: Dataset entry with 'ground_truth' and 'options' fields
        pred: Prediction string (e.g., "AC")
    
    Returns:
        Tuple of (exactly_correct, num_correct_options, cleaned_pred)
        - exactly_correct: 1 if prediction exactly matches ground truth, 0 otherwise
        - num_correct_options: Number of correct options (count, not fraction)
        - cleaned_pred: The cleaned and validated prediction string
    
    Raises:
        InvalidPredictionError: If prediction format is invalid (not pure A-Z or has duplicates)
    """
    # Clean up prediction (remove any trailing reasoning tags)
    if (idx := pred.rfind("</think>")) != -1:
        pred = pred[idx + len("</think>"):]
    
    pred = pred.strip()
    
    # Validate prediction format
    if not pred:
        raise InvalidPredictionError(f"Prediction is empty: {pred!r}")
    
    if not all(c.isalpha() and c.isupper() for c in pred):
        raise InvalidPredictionError(f"Prediction contains non-A-Z characters: {pred!r}")
    
    if len(set(pred)) != len(pred):  # duplicate options
        raise InvalidPredictionError(f"Prediction contains duplicate options: {pred!r}")
    
    # Check that all selected options are within valid range
    # Get number of options from entry
    num_options = len(entry["options"])
    max_valid_letter = chr(ord("A") + num_options - 1)  # e.g., if num_options=3, max is "C"
    for c in pred:
        if c > max_valid_letter:
            raise InvalidPredictionError(
                f"Prediction contains option '{c}' which is out of range. "
                f"Valid options are A-{max_valid_letter} (only {num_options} options available): {pred!r}"
            )
    
    # Get ground truth (num_options already set above)
    ground_truth = entry["ground_truth"]
    
    # Convert ground_truth to string if it's a set or list
    # Handle both integer indices (e.g., [0, 1]) and string letters (e.g., ["A", "B"])
    if isinstance(ground_truth, (set, list)):
        if ground_truth:
            # Check if first element is an integer (indices) or string (letters)
            if isinstance(next(iter(ground_truth)), int):
                # Convert integer indices to letters
                ground_truth_str = "".join(sorted(index_to_letter(i) for i in ground_truth))
                ground_truth_set = set(index_to_letter(i) for i in ground_truth)
            else:
                # Already strings, just join them
                ground_truth_str = "".join(sorted(ground_truth))
                ground_truth_set = set(ground_truth) if isinstance(ground_truth, set) else set(ground_truth)
        else:
            # Empty list/set
            ground_truth_str = ""
            ground_truth_set = set()
    elif isinstance(ground_truth, int):
        # Single integer index
        ground_truth_str = index_to_letter(ground_truth)
        ground_truth_set = {ground_truth_str}
    else:
        # Already a string
        ground_truth_str = ground_truth
        ground_truth_set = set(ground_truth_str)
    
    # Sort prediction for comparison (metrics.py compares strings)
    pred_sorted = "".join(sorted(pred))
    pred_set = set(pred)
    
    # Calculate per-question accuracy (exactly correct) using metrics.py
    exactly_correct = calculate_per_question_accuracy(ground_truth_str, pred_sorted)
    
    # Return count of correctly selected options (intersection size)
    # This is used for detailed results tracking
    # Note: Per-option accuracy fraction is calculated separately in the evaluation loop
    # using calculate_per_option_accuracy from metrics.py
    num_correct_options = len(pred_set & ground_truth_set)
    
    return exactly_correct, num_correct_options, pred


def sanitize_prompt(prompt: List[Dict[str, Any]], video_id: str) -> List[Dict[str, Any]]:
    """
    Create a sanitized version of the prompt with video binary data removed.
    
    Replaces base64-encoded video data with a reference to the video file.
    
    Args:
        prompt: Original prompt (list of message dictionaries)
        video_id: Video identifier for reference
    
    Returns:
        Sanitized prompt with video binary data replaced by reference
    """
    sanitized_prompt = []
    for message in prompt:
        sanitized_message = message.copy()
        if "content" in sanitized_message:
            sanitized_content = []
            for item in sanitized_message["content"]:
                if item.get("type") == "video_url":
                    # Replace base64 video data with a reference
                    sanitized_content.append({
                        "type": "video_url",
                        "video_url": {"url": f"<video_reference:{video_id}>"},
                    })
                else:
                    # Keep other content as-is (e.g., text)
                    sanitized_content.append(item)
            sanitized_message["content"] = sanitized_content
        sanitized_prompt.append(sanitized_message)
    return sanitized_prompt


class AsyncEvaluator:
    """Async evaluator with retry logic."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
        fps: int = 5,
    ):
        """
        Initialize async evaluator.
        
        Args:
            base_url: Base URL for the API
            model: Model name
            api_key: API key (default: "EMPTY")
            max_tokens: Maximum tokens for generation (default: from model config)
            temperature: Sampling temperature (default: from model config)
            extra_body: Extra body parameters for API (merged with model defaults)
            max_retries: Maximum number of retries
            fps: Frames per second for video processing (default: 5)
        """
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.fps = fps
        
        # Get model-specific hyperparameters
        model_hparams = get_model_hyperparameters(model)
        
        # Use provided values or fall back to model defaults
        self.max_tokens = max_tokens if max_tokens is not None else model_hparams.get("max_tokens")
        self.temperature = temperature if temperature is not None else model_hparams.get("temperature")
        
        # Build extra_body: only include keys that are present in model_hparams or extra_body
        # Don't include keys that are missing or None
        extra_body_dict = {
            "mm_processor_kwargs": {"fps": fps, "do_sample_frames": True,}
        }
        # "video": {"size": {"shortest_edge": 384 * 240, "longest_edge": 384 * 240 * 500}}
        # do_size: False
        # these all doesn't quite work to specify the resize behavior
        
        # Add keys from model_hparams if they exist and are not None
        for key in ["top_k", "top_p", "repetition_penalty", "presence_penalty"]:
            if key in model_hparams and model_hparams[key] is not None:
                extra_body_dict[key] = model_hparams[key]
        
        # Merge with provided extra_body, only including non-None values
        if extra_body:
            for key, value in extra_body.items():
                if value is not None:
                    extra_body_dict[key] = value
        
        self.extra_body = extra_body_dict if extra_body_dict else None
        
        self.max_retries = max_retries
        # Track which entries have had their first error printed
        self._first_error_printed = set()
    
    def _generate_random_prediction(self, entry: Dict[str, Any]) -> str:
        """
        Generate a random prediction for an entry.
        
        Randomly selects uniformly from all possible combinations of exactly 2 options.
        
        Args:
            entry: Dataset entry with 'ground_truth' and 'options' fields
        
        Returns:
            Prediction string in valid format (e.g., "AC" for a pair)
        
        Raises:
            ValueError: If there are fewer than 2 options available
        """
        num_available_options = len(entry["options"])
        
        # Check if we have at least 2 options
        if num_available_options < 2:
            raise ValueError(f"Need at least 2 options, but only {num_available_options} available")
        
        # Generate all possible combinations of exactly 2 options
        # For n options, we have C(n,2) = n*(n-1)/2 possible pairs
        all_combinations = list(itertools.combinations(range(num_available_options), 2))
        
        # Randomly select one combination of exactly 2 options
        selected_indices = random.choice(all_combinations)
        
        # Convert to letters (A-Z), sorted for consistency
        prediction = "".join(chr(ord("A") + idx) for idx in sorted(selected_indices))
        
        return prediction
    
    def _make_print_first_retry_callback(self, entry: Dict[str, Any]):
        """Create a callback that prints error on first retry for this specific entry."""
        entry_id = entry.get('video', 'unknown')
        
        def print_first_retry(retry_state: RetryCallState):
            if retry_state.attempt_number == 2:  # First retry (attempt 2 = second attempt)
                if entry_id not in self._first_error_printed:
                    exception = retry_state.outcome.exception()
                    print(f"Error on first retry for entry {entry_id}: {exception}")
                    self._first_error_printed.add(entry_id)
        
        return print_first_retry
    
    async def evaluate_entry(
        self,
        entry: Dict[str, Any],
        dataset_name: str,
        base_dir: str = ".",
        video_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single entry with retry logic.
        
        Validates that predictions are pure A-Z with no duplicates, and retries
        if the format is invalid.
        
        Args:
            entry: Dataset entry
            dataset_name: Name of the dataset
            base_dir: Base directory for the project
            video_filename: Optional specific video filename to use (if None, uses first available)
        
        Returns:
            Dictionary with keys:
            - 'prediction': Prediction string (guaranteed to be valid format: pure A-Z, no duplicates)
            - 'prompt': List of message dictionaries (the exact prompt sent to the model)
            - 'response': Raw response string from the model (None for random baseline)
            - 'token_usage': Dictionary with 'prompt_tokens', 'completion_tokens', 'total_tokens' (None for random baseline)
            - 'video_filename': The video filename used for this evaluation
        
        Raises:
            BadRequestError: If the video file is invalid/corrupted (not retried)
            InvalidPredictionError: If prediction format is invalid after max retries
            Other exceptions: Retried up to 10 times
        """
        # Build prompt (needed for both random and API-based evaluation)
        # Add an explicit hint about the number of correct options
        num_correct = _num_correct_options_from_entry(entry)
        hint_text = f"Hint: There are exactly {num_correct} correct options."
        messages = build_prompt(
            entry,
            dataset_name,
            base_dir,
            additional_suffix=hint_text,
            video_filename=video_filename,
        )
        
        # Handle random baseline
        if self.model == "random":
            # For random baseline, generate prediction without API call
            pred = self._generate_random_prediction(entry)
            # Validate the prediction format (should always be valid, but check anyway)
            _, _, cleaned_pred = validate_and_get_metrics(entry, pred)
            # Get video filename if not provided
            if video_filename is None:
                available_videos = get_available_videos(entry, dataset_name, base_dir)
                video_filename = available_videos[0] if available_videos else None
            return {
                'prediction': cleaned_pred,
                'prompt': messages,
                'response': None,  # No API response for random baseline
                'token_usage': None,  # No token usage for random baseline
                'video_filename': video_filename,
            }
        
        # Create retry decorator with entry-specific callback
        @retry(
            stop=stop_after_attempt(10),
            wait=wait_none(),
            # Retry on InvalidPredictionError, API errors, but NOT on BadRequestError or other ValueError
            # BadRequestError indicates invalid input (e.g., corrupted video), which won't be fixed by retrying
            # InvalidPredictionError indicates invalid prediction format, which should be retried
            # Note: InvalidPredictionError is explicitly included even though it's a ValueError subclass
            retry=_retry_condition,
            before_sleep=self._make_print_first_retry_callback(entry),
            reraise=True,
        )
        async def _evaluate_with_retry():
            # Initialize video_filename from outer scope to avoid UnboundLocalError on retry
            current_video_filename = video_filename
            try:
                # Build request kwargs, only including extra_body if it has values
                request_kwargs = {
                    "messages": messages,
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                if self.extra_body:
                    request_kwargs["extra_body"] = self.extra_body
                
                response = await self.client.chat.completions.create(**request_kwargs)
                
                raw_response = response.choices[0].message.content
                if raw_response is None:
                    raise ValueError("Empty response from model")
                
                # Extract token usage from response
                token_usage = None
                if hasattr(response, 'usage') and response.usage is not None:
                    token_usage = {
                        'prompt_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else None,
                        'completion_tokens': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else None,
                        'total_tokens': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else None,
                    }
                
                pred = raw_response.strip()
                # Validate and clean prediction format - this will raise
                # InvalidPredictionError if invalid, which will trigger a retry
                _, _, cleaned_pred = validate_and_get_metrics(entry, pred)
                
                # Get video filename if not provided
                if current_video_filename is None:
                    available_videos = get_available_videos(entry, dataset_name, base_dir)
                    current_video_filename = available_videos[0] if available_videos else None
                
                return {
                    # Store the cleaned prediction (letters only) for downstream metrics/logging
                    'prediction': cleaned_pred,
                    'prompt': messages,
                    'response': raw_response,  # Include original response before stripping
                    'token_usage': token_usage,  # Token usage information
                    'video_filename': current_video_filename,
                }
            
            except BadRequestError as e:
                # BadRequestError (400) indicates invalid input - don't retry, just raise
                # This typically means the video file is corrupted or invalid
                print(f"BadRequestError (likely corrupted video) for entry {entry.get('video', 'unknown')}: {e}")
                raise
            except InvalidPredictionError:
                # InvalidPredictionError will be retried by the decorator
                raise
            except Exception as e:
                # Other errors will be retried, error printed on first retry via before_sleep callback
                raise
        
        return await _evaluate_with_retry()
    
    async def evaluate_entry_with_samples(
        self,
        entry: Dict[str, Any],
        dataset_name: str,
        num_samples: int,
        base_dir: str = ".",
    ) -> List[Dict[str, Any]]:
        """
        Evaluate an entry multiple times (sampling).
        
        Evaluates all available camera angles, with num_samples repeats per camera angle.
        For example: 4 camera angles with num_samples=4 = 16 total evaluations.
        
        Args:
            entry: Dataset entry
            dataset_name: Name of the dataset
            num_samples: Number of samples per camera angle (prompt-level sampling)
            base_dir: Base directory for the project
        
        Returns:
            List of dictionaries, each with keys:
            - 'prediction': Prediction string
            - 'prompt': List of message dictionaries (the exact prompt sent to the model)
            - 'response': Raw response string from the model (None for random baseline)
            - 'token_usage': Dictionary with 'prompt_tokens', 'completion_tokens', 'total_tokens' (None for random baseline)
            - 'video_filename': The video filename used for this evaluation
        """
        # Get all available camera angles
        try:
            available_videos = get_available_videos(entry, dataset_name, base_dir)
        except FileNotFoundError:
            # Fallback: if no videos found, use legacy behavior with random selection
            available_videos = []
        
        if not available_videos:
            # Legacy behavior: random selection for each sample
            tasks = [
                self.evaluate_entry(entry, dataset_name, base_dir)
                for _ in range(num_samples)
            ]
        else:
            # For each camera angle, evaluate it num_samples times
            tasks = [
                self.evaluate_entry(entry, dataset_name, base_dir, video_filename=video_filename)
                for video_filename in available_videos
                for _ in range(num_samples)
            ]
        
        return await asyncio.gather(*tasks)


async def evaluate_dataset(
    entries: List[Dict[str, Any]],
    dataset_name: str,
    evaluator: AsyncEvaluator,
    num_samples: int = 1,
    max_concurrent: int = 10,
    base_dir: str = ".",
    max_entries: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate entire dataset asynchronously.
    
    Args:
        entries: List of dataset entries to evaluate
        dataset_name: Name of the dataset
        evaluator: AsyncEvaluator instance
        num_samples: Number of samples per question
        max_concurrent: Maximum concurrent requests
        base_dir: Base directory for the project
        max_entries: Maximum number of entries to evaluate (applied after loading all files)
    
    Returns:
        Dictionary with evaluation results and metrics
    """
    print(f"Total entries loaded: {len(entries)}")
    
    if max_entries is not None:
        entries = entries[:max_entries]
        print(f"Limited to {len(entries)} entries (max_entries={max_entries})")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate entry with semaphore for concurrency control."""
        async with semaphore:
            try:
                predictions = await evaluator.evaluate_entry_with_samples(
                    entry, dataset_name, num_samples, base_dir
                )
                return {
                    "entry": entry,
                    "predictions": predictions,
                    "error": None,
                }
            except Exception as e:
                return {
                    "entry": entry,
                    "predictions": [],
                    "error": str(e),
                }
    
    # Evaluate all entries
    print(f"Evaluating {len(entries)} entries with {num_samples} sample(s) per camera angle...")
    
    # Create tasks with indices
    async def evaluate_with_idx(idx: int, entry: Dict[str, Any]) -> Dict[str, Any]:
        result = await evaluate_with_semaphore(entry)
        result["entry_idx"] = idx
        return result
    
    # Create all tasks at once - semaphore will control concurrency
    tasks = [
        asyncio.create_task(evaluate_with_idx(idx, entry))
        for idx, entry in enumerate(entries)
    ]
    
    results = []
    
    with tqdm(total=len(entries), desc="Evaluating") as pbar:
        try:
            # Process tasks as they complete
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    results.append(result)
                except Exception as e:
                    # This shouldn't happen due to exception handling in evaluate_with_semaphore
                    # but handle it just in case
                    print(f"Unexpected exception: {e}")
                finally:
                    pbar.update(1)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted! Cancelling remaining tasks...")
            # Cancel all pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellations to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            print(f"Cancelled tasks. Saving {len(results)} partial results...")
            raise
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x["entry_idx"])
    
    # Calculate metrics
    total_questions = len(entries)
    total_samples = total_questions * num_samples
    
    # Per-question metrics: a question is correct if ANY sample is exactly correct
    question_exactly_correct = 0  # @k (where k=num_samples): any sample correct
    question_first_sample_correct = 0  # @1: first sample only
    
    # Per-option metrics: aggregate across all samples using metrics.py
    # Accumulate per-option accuracy fractions (from calculate_per_option_accuracy)
    total_per_option_acc = 0.0
    total_samples_for_per_option = 0
    
    # Token usage tracking
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    samples_with_token_usage = 0
    
    # Per-question-type metrics
    question_type_stats = {}  # question_type -> {total, exactly_correct, first_sample_correct, per_option_acc_sum, per_option_samples}
    
    # Detailed results
    detailed_results = []
    
    for result in results:
        entry = result["entry"]
        entry_idx = result["entry_idx"]
        predictions = result["predictions"]
        error = result["error"]
        
        # Extract question type from metadata
        question_type = entry.get("metadata", {}).get("question_type", "unknown")
        
        if error:
            # For errored entries, record a consistent structure with empty per-sample fields
            detailed_results.append({
                "entry_idx": entry_idx,
                "video": entry.get("video"),
                "question_type": question_type,
                "ground_truth": entry.get("ground_truth"),
                "predictions": [],
                "prompts": [],
                "responses": [],
                "token_usage": [],
                "sample_metrics": [],
                "question_has_exact_match": False,
                "question_total_correct_options": 0,
                "first_sample_correct": False,
                "error": error,
            })
            # Still track question type stats even for errors (count as total)
            if question_type not in question_type_stats:
                question_type_stats[question_type] = {
                    "total": 0,
                    "exactly_correct": 0,
                    "first_sample_correct": 0,
                    "per_option_acc_sum": 0.0,
                    "per_option_samples": 0,
                }
            question_type_stats[question_type]["total"] += 1
            continue
        
        # Normalize ground_truth to a set of strings (letters)
        # Handle both integer indices (e.g., [0, 1]) and string letters (e.g., ["A", "B"])
        raw_ground_truth = entry["ground_truth"]
        if isinstance(raw_ground_truth, (set, list)):
            if raw_ground_truth:
                # Check if first element is an integer (indices) or string (letters)
                if isinstance(next(iter(raw_ground_truth)), int):
                    # Convert integer indices to letters
                    ground_truth = set(index_to_letter(i) for i in raw_ground_truth)
                else:
                    # Already strings
                    ground_truth = set(raw_ground_truth)
            else:
                # Empty list/set
                ground_truth = set()
        else:
            # Single value (string or int)
            if isinstance(raw_ground_truth, int):
                ground_truth = {index_to_letter(raw_ground_truth)}
            else:
                ground_truth = set(raw_ground_truth)
        
        num_options = len(entry["options"])
        
        # Initialize question type stats if needed
        if question_type not in question_type_stats:
            question_type_stats[question_type] = {
                "total": 0,
                "exactly_correct": 0,
                "first_sample_correct": 0,
                "per_option_acc_sum": 0.0,
                "per_option_samples": 0,
            }
        question_type_stats[question_type]["total"] += 1
        
        # Calculate metrics for each prediction
        # predictions is now a list of dicts with keys: 'prediction', 'prompt', 'response'
        sample_metrics = []
        question_has_exact_match = False
        question_total_correct = 0
        first_sample_correct = False
        
        # Extract prompt from first sample (should be the same for all samples)
        # Store prompts and responses for each sample
        sample_prompts = []
        sample_responses = []
        sample_token_usage = []
        prediction_strings = []
        
        for idx, pred_dict in enumerate(predictions):
            # Extract prediction string from dictionary
            pred = pred_dict['prediction']
            prompt = pred_dict['prompt']
            response = pred_dict['response']
            token_usage = pred_dict.get('token_usage')
            
            # Sanitize prompt to remove video binary data before storing
            video_id = entry.get("video", "unknown")
            sanitized_prompt = sanitize_prompt(prompt, video_id)
            
            # Store sanitized prompt, response, and token usage for this sample
            sample_prompts.append(sanitized_prompt)
            sample_responses.append(response)
            sample_token_usage.append(token_usage)
            
            try:
                exactly_correct, num_correct, pred_cleaned = validate_and_get_metrics(entry, pred)
                # Also calculate per-option accuracy using metrics.py
                # Use the cleaned prediction from validate_and_get_metrics
                ground_truth_str = "".join(sorted(ground_truth))
                pred_sorted = "".join(sorted(pred_cleaned))
                per_option_acc = calculate_per_option_accuracy(num_options, ground_truth_str, pred_sorted)
            except InvalidPredictionError as e:
                # This shouldn't happen since we validate in evaluate_entry,
                # but handle it defensively
                print(f"Warning: Invalid prediction in results for entry {entry.get('video', 'unknown')}: {e}")
                # Treat as incorrect
                exactly_correct, num_correct = 0, 0
                per_option_acc = 0.0
            
            # Use the cleaned prediction for all logging/metrics
            prediction_strings.append(pred_cleaned)
            sample_metrics.append({
                "prediction": pred_cleaned,
                "exactly_correct": exactly_correct,
                "num_correct_options": num_correct,
                "per_option_accuracy": per_option_acc,
            })
            
            # @1: Check if first sample is correct
            if idx == 0 and exactly_correct:
                first_sample_correct = True
            
            # @k: Check if any sample is correct
            if exactly_correct:
                question_has_exact_match = True
            
            question_total_correct += num_correct
            # Accumulate per-option accuracy fractions
            total_per_option_acc += per_option_acc
            total_samples_for_per_option += 1
            # Also track for question type stats (using fraction)
            if "per_option_acc_sum" not in question_type_stats[question_type]:
                question_type_stats[question_type]["per_option_acc_sum"] = 0.0
                question_type_stats[question_type]["per_option_samples"] = 0
            question_type_stats[question_type]["per_option_acc_sum"] += per_option_acc
            question_type_stats[question_type]["per_option_samples"] += 1
            
            # Aggregate token usage
            if token_usage is not None:
                if token_usage.get('prompt_tokens') is not None:
                    total_prompt_tokens += token_usage['prompt_tokens']
                if token_usage.get('completion_tokens') is not None:
                    total_completion_tokens += token_usage['completion_tokens']
                if token_usage.get('total_tokens') is not None:
                    total_tokens += token_usage['total_tokens']
                samples_with_token_usage += 1
        
        if question_has_exact_match:
            question_exactly_correct += 1
            question_type_stats[question_type]["exactly_correct"] += 1
        
        if first_sample_correct:
            question_first_sample_correct += 1
            question_type_stats[question_type]["first_sample_correct"] += 1
        
        detailed_results.append({
            "entry_idx": entry_idx,
            "video": entry.get("video"),
            "question_type": question_type,
            "ground_truth": entry["ground_truth"],
            "predictions": prediction_strings,  # Keep as list of strings for backward compatibility
            "prompts": sample_prompts,  # List of prompts (one per sample)
            "responses": sample_responses,  # List of responses (one per sample)
            "token_usage": sample_token_usage,  # List of token usage dicts (one per sample)
            "sample_metrics": sample_metrics,
            "question_has_exact_match": question_has_exact_match,
            "question_total_correct_options": question_total_correct,
            "first_sample_correct": first_sample_correct,
        })
    
    # Calculate final metrics
    per_question_accuracy = question_exactly_correct / total_questions if total_questions > 0 else 0.0
    # Per-option accuracy: average of per-sample fractions from calculate_per_option_accuracy
    per_option_accuracy = total_per_option_acc / total_samples_for_per_option if total_samples_for_per_option > 0 else 0.0
    
    # Calculate @1 and @k metrics
    accuracy_at_1 = question_first_sample_correct / total_questions if total_questions > 0 else 0.0
    accuracy_at_k = question_exactly_correct / total_questions if total_questions > 0 else 0.0
    
    metrics_dict = {
        "per_question_accuracy": per_question_accuracy,
        "per_option_accuracy": per_option_accuracy,
        "questions_with_exact_match": question_exactly_correct,
        "total_samples": total_samples_for_per_option,
        "token_usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "samples_with_token_usage": samples_with_token_usage,
        },
    }
    
    # Add @1 and @k metrics when num_samples > 1
    if num_samples > 1:
        metrics_dict["accuracy@1"] = accuracy_at_1
        metrics_dict[f"accuracy@{num_samples}"] = accuracy_at_k
        metrics_dict["questions_with_first_sample_correct"] = question_first_sample_correct
        metrics_dict["questions_with_any_sample_correct"] = question_exactly_correct
    
    # Calculate per-question-type metrics
    per_question_type_metrics = {}
    for qtype, stats in question_type_stats.items():
        qtype_total = stats["total"]
        if qtype_total > 0:
            qtype_per_option_acc = stats["per_option_acc_sum"] / stats["per_option_samples"] if stats["per_option_samples"] > 0 else 0.0
            qtype_metrics = {
                "total_questions": qtype_total,
                "per_question_accuracy": stats["exactly_correct"] / qtype_total,
                "per_option_accuracy": qtype_per_option_acc,
                "questions_with_exact_match": stats["exactly_correct"],
                "total_samples": stats["per_option_samples"],
            }
            if num_samples > 1:
                qtype_metrics["accuracy@1"] = stats["first_sample_correct"] / qtype_total
                qtype_metrics[f"accuracy@{num_samples}"] = stats["exactly_correct"] / qtype_total
                qtype_metrics["questions_with_first_sample_correct"] = stats["first_sample_correct"]
                qtype_metrics["questions_with_any_sample_correct"] = stats["exactly_correct"]
            per_question_type_metrics[qtype] = qtype_metrics
    
    metrics_dict["per_question_type"] = per_question_type_metrics
    
    return {
        "model": evaluator.model,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "fps": evaluator.fps,
        "total_questions": total_questions,
        "total_samples": total_samples,
        "metrics": metrics_dict,
        "detailed_results": detailed_results,
    }


def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    # Don't use custom signal handlers with asyncio - they conflict with the event loop
    # Instead, let asyncio handle KeyboardInterrupt naturally
    pass


async def main():
    """Main async function."""
    setup_signal_handlers()
    
    parser = argparse.ArgumentParser(description="Async evaluation script for causal pool dataset")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., '1k_simple')",
    )
    parser.add_argument(
        "-u", "--base-url",
        type=str,
        default="http://trig0002:8000/v1",
        help="Base URL for the API",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'Qwen/Qwen3-VL-4B-Instruct')",
    )
    parser.add_argument(
        "-k", "--api-key",
        type=str,
        default="EMPTY",
        help="API key (default: 'EMPTY')",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=1,
        help="Number of samples per question (default: 1)",
    )
    parser.add_argument(
        "-c", "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)",
    )
    parser.add_argument(
        "-t", "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for generation (default: None)",
    )
    parser.add_argument(
        "-T", "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: None)",
    )
    parser.add_argument(
        "-b", "--base-dir",
        type=str,
        default=".",
        help="Base directory for the project (default: current directory)",
    )
    parser.add_argument(
        "-e", "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to evaluate (for testing, default: all)",
    )
    parser.add_argument(
        "-C", "--counterfactual-velocity-size",
        type=int,
        default=256,
        help="Number of entries to load from test-counterfactual_velocity.jsonl (default: 256, use -1 for all)",
    )
    parser.add_argument(
        "--counterfactual-position-size",
        type=int,
        default=256,
        help="Number of entries to load from test-counterfactual_position.jsonl (default: 256, use -1 for all)",
    )
    parser.add_argument(
        "-D", "--descriptive-size",
        type=int,
        default=256,
        help="Number of entries to load from test-descriptive.jsonl (default: 256, use -1 for all)",
    )
    parser.add_argument(
        "-P", "--predictive-size",
        type=int,
        default=256,
        help="Number of entries to load from test-predictive.jsonl (default: 256, use -1 for all)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for video processing (default: 5)",
    )
    
    args = parser.parse_args()
    
    # Build sizes dict for gather_test_dataset
    # Keys must match: counterfactual_velocity, counterfactual_position, descriptive, predictive
    sizes = {}
    
    # Check if files exist and get sizes
    dataset_splits_dir = os.path.join(args.base_dir, "datasets", args.dataset, "splits")
    
    # Counterfactual velocity
    counterfactual_velocity_path = os.path.join(dataset_splits_dir, "test-counterfactual_velocity.jsonl")
    if not os.path.exists(counterfactual_velocity_path):
        raise FileNotFoundError(f"Dataset file not found: {counterfactual_velocity_path}")
    sizes["counterfactual_velocity"] = args.counterfactual_velocity_size
    
    # Counterfactual position
    counterfactual_position_path = os.path.join(dataset_splits_dir, "test-counterfactual_position.jsonl")
    if not os.path.exists(counterfactual_position_path):
        raise FileNotFoundError(f"Dataset file not found: {counterfactual_position_path}")
    sizes["counterfactual_position"] = args.counterfactual_position_size
    
    # Descriptive
    descriptive_path = os.path.join(dataset_splits_dir, "test-descriptive.jsonl")
    if not os.path.exists(descriptive_path):
        raise FileNotFoundError(f"Dataset file not found: {descriptive_path}")
    sizes["descriptive"] = args.descriptive_size
    
    # Predictive (optional)
    if args.predictive_size > 0 or args.predictive_size == -1:
        predictive_path = os.path.join(dataset_splits_dir, "test-predictive.jsonl")
        if not os.path.exists(predictive_path):
            raise FileNotFoundError(f"Dataset file not found: {predictive_path}")
        sizes["predictive"] = args.predictive_size
    
    # Load dataset using gather_test_dataset
    # Note: gather_test_dataset uses relative paths, so we need to change to base_dir
    original_cwd = os.getcwd()
    try:
        os.chdir(args.base_dir)
        print(f"Loading datasets with sizes: {sizes}")
        dataset = gather_test_dataset(args.dataset, sizes, random_seed=42)
    finally:
        os.chdir(original_cwd)
    
    # Convert Dataset to list of entries
    entries = [entry for entry in dataset]
    print(f"Loaded {len(entries)} total entries")
    
    # Handle random baseline
    if args.model == "random":
        # For random baseline, we don't need API or hyperparameters
        print("Using random baseline - predictions will be randomly generated")
        # Set a seed for reproducibility (optional, but helpful)
        random.seed(42)
    else:
        # Get model-specific hyperparameters for display
        model_hparams = get_model_hyperparameters(args.model)
        
        # Resolve hyperparameters: use provided values or model defaults
        max_tokens = args.max_tokens if args.max_tokens is not None else model_hparams.get("max_tokens")
        temperature = args.temperature if args.temperature is not None else model_hparams.get("temperature")
    
    # Create evaluator
    evaluator = AsyncEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        fps=args.fps,
    )
    
    # Evaluate dataset
    print(f"Starting evaluation...")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Samples per question: {args.num_samples}")
    if args.model != "random":
        print(f"  Max concurrent: {args.max_concurrent}")
        print(f"  Hyperparameters:")
        model_hparams = get_model_hyperparameters(args.model)
        max_tokens = args.max_tokens if args.max_tokens is not None else model_hparams.get("max_tokens")
        temperature = args.temperature if args.temperature is not None else model_hparams.get("temperature")
        if max_tokens is not None:
            print(f"    max_tokens: {max_tokens}" + (" (from model config)" if args.max_tokens is None else " (override)"))
        if temperature is not None:
            print(f"    temperature: {temperature}" + (" (from model config)" if args.temperature is None else " (override)"))
        # Only print sampling parameters that exist in model config
        for key in ["top_k", "top_p", "repetition_penalty", "presence_penalty"]:
            if key in model_hparams and model_hparams[key] is not None:
                print(f"    {key}: {model_hparams[key]} (from model config)")
    else:
        print(f"  Max concurrent: {args.max_concurrent} (not used for random baseline)")
    
    # Warn about very high concurrency (only for non-random models)
    if args.model != "random" and args.max_concurrent > 100:
        print(f"\n⚠️  WARNING: Very high concurrency ({args.max_concurrent}) may cause:")
        print("   - Memory issues")
        print("   - Connection pool exhaustion")
        print("   Consider using a lower value (e.g., 50-100) for better stability.\n")
    
    try:
        results = await evaluate_dataset(
            entries=entries,
            dataset_name=args.dataset,
            evaluator=evaluator,
            num_samples=args.num_samples,
            max_concurrent=args.max_concurrent,
            base_dir=args.base_dir,
            max_entries=args.max_entries,
        )
        
        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Per-question accuracy: {results['metrics']['per_question_accuracy']:.4f}")
        print(f"Per-option accuracy: {results['metrics']['per_option_accuracy']:.4f}")
        print(f"Questions with exact match: {results['metrics']['questions_with_exact_match']}/{results['total_questions']}")
        print(f"Total samples: {results['total_samples']}")
        
        # Print token usage if available
        if 'token_usage' in results['metrics']:
            token_usage = results['metrics']['token_usage']
            if token_usage['samples_with_token_usage'] > 0:
                print(f"\nToken Usage:")
                print(f"  Total prompt tokens: {token_usage['total_prompt_tokens']:,}")
                print(f"  Total completion tokens: {token_usage['total_completion_tokens']:,}")
                print(f"  Total tokens: {token_usage['total_tokens']:,}")
                print(f"  Samples with token usage: {token_usage['samples_with_token_usage']}/{results['total_samples']}")
                if token_usage['samples_with_token_usage'] > 0:
                    avg_prompt = token_usage['total_prompt_tokens'] / token_usage['samples_with_token_usage']
                    avg_completion = token_usage['total_completion_tokens'] / token_usage['samples_with_token_usage']
                    avg_total = token_usage['total_tokens'] / token_usage['samples_with_token_usage']
                    print(f"  Average per sample:")
                    print(f"    Prompt tokens: {avg_prompt:.1f}")
                    print(f"    Completion tokens: {avg_completion:.1f}")
                    print(f"    Total tokens: {avg_total:.1f}")
        
        # Print @1 and @k metrics when num_samples > 1
        if results['num_samples'] > 1:
            print("\n@k Metrics:")
            # @1: first sample only
            if "accuracy@1" in results['metrics']:
                count = results['metrics']['questions_with_first_sample_correct']
                accuracy = results['metrics']['accuracy@1']
                print(f"  Accuracy@1 (first sample only): {accuracy:.4f} ({count}/{results['total_questions']} questions)")
            # @k: any sample correct (where k = num_samples)
            accuracy_key = f"accuracy@{results['num_samples']}"
            if accuracy_key in results['metrics']:
                count = results['metrics']['questions_with_any_sample_correct']
                accuracy = results['metrics'][accuracy_key]
                print(f"  Accuracy@{results['num_samples']} (any sample correct): {accuracy:.4f} ({count}/{results['total_questions']} questions)")
        
        # Print per-question-type metrics
        if "per_question_type" in results['metrics'] and results['metrics']['per_question_type']:
            print("\nPer-Question-Type Metrics:")
            for qtype, qtype_metrics in sorted(results['metrics']['per_question_type'].items()):
                print(f"\n  {qtype}:")
                print(f"    Total questions: {qtype_metrics['total_questions']}")
                print(f"    Per-question accuracy: {qtype_metrics['per_question_accuracy']:.4f}")
                print(f"    Per-option accuracy: {qtype_metrics['per_option_accuracy']:.4f}")
                print(f"    Questions with exact match: {qtype_metrics['questions_with_exact_match']}/{qtype_metrics['total_questions']}")
                if results['num_samples'] > 1:
                    if "accuracy@1" in qtype_metrics:
                        print(f"    Accuracy@1: {qtype_metrics['accuracy@1']:.4f} ({qtype_metrics['questions_with_first_sample_correct']}/{qtype_metrics['total_questions']} questions)")
                    accuracy_key = f"accuracy@{results['num_samples']}"
                    if accuracy_key in qtype_metrics:
                        print(f"    Accuracy@{results['num_samples']}: {qtype_metrics[accuracy_key]:.4f} ({qtype_metrics['questions_with_any_sample_correct']}/{qtype_metrics['total_questions']} questions)")
        
        print("=" * 50)
        
        # Save results
        normalized_model = normalize_model_name(args.model)
        output_path = os.path.join(
            args.base_dir,
            "results",
            args.dataset,
            f"eval_{normalized_model}.json",
        )
        save_results(results, output_path)
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user. Attempting to save partial results...")
        # Note: The evaluate_dataset function re-raises KeyboardInterrupt after setting up partial results
        raise


if __name__ == "__main__":
    try:
        # Use asyncio.run with proper event loop policy for better signal handling
        if sys.platform != 'win32':
            # On Unix, use the default event loop policy
            asyncio.run(main())
        else:
            # On Windows, use ProactorEventLoop
            asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
