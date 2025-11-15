#!/usr/bin/env python3
"""
Async evaluation script for causal pool dataset.

This script evaluates a model on a dataset with support for:
- Async concurrent evaluation
- Retry logic with tenacity
- Multiple samples per question
- Per-question and per-option accuracy metrics
"""

import argparse
import asyncio
import json
import os
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
    get_metrics,
    build_prompt,
)
from causal_pool.data.dataset_utils import gather_test_dataset
from causal_pool.utils import normalize_model_name


# Create retry condition by combining existing helpers
# Retry on InvalidPredictionError OR API errors OR (everything except BadRequestError and ValueError)
# Note: InvalidPredictionError is explicitly included even though it's a ValueError subclass
_retry_condition = (
    retry_if_exception_type((InvalidPredictionError, APIError, APIConnectionError, APITimeoutError))
    | retry_if_not_exception_type((BadRequestError, ValueError))
)


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
        """
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        
        # Get model-specific hyperparameters
        model_hparams = get_model_hyperparameters(model)
        
        # Use provided values or fall back to model defaults
        self.max_tokens = max_tokens if max_tokens is not None else model_hparams.get("max_tokens")
        self.temperature = temperature if temperature is not None else model_hparams.get("temperature")
        
        # Build extra_body: only include keys that are present in model_hparams or extra_body
        # Don't include keys that are missing or None
        extra_body_dict = {}
        
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
        
        Randomly selects the same number of options as the ground truth,
        from the available options.
        
        Args:
            entry: Dataset entry with 'ground_truth' and 'options' fields
        
        Returns:
            Prediction string in valid format (e.g., "AC")
        """
        ground_truth = entry["ground_truth"]
        num_options_to_select = len(ground_truth)
        num_available_options = len(entry["options"])
        
        # Randomly select indices
        selected_indices = sorted(random.sample(range(num_available_options), num_options_to_select))
        
        # Convert to letters (A-Z)
        prediction = "".join(chr(ord("A") + idx) for idx in selected_indices)
        
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
    ) -> str:
        """
        Evaluate a single entry with retry logic.
        
        Validates that predictions are pure A-Z with no duplicates, and retries
        if the format is invalid.
        
        Args:
            entry: Dataset entry
            dataset_name: Name of the dataset
            base_dir: Base directory for the project
        
        Returns:
            Prediction string (guaranteed to be valid format: pure A-Z, no duplicates)
        
        Raises:
            BadRequestError: If the video file is invalid/corrupted (not retried)
            InvalidPredictionError: If prediction format is invalid after max retries
            Other exceptions: Retried up to 10 times
        """
        # Handle random baseline
        if self.model == "random":
            # For random baseline, generate prediction without API call
            pred = self._generate_random_prediction(entry)
            # Validate the prediction format (should always be valid, but check anyway)
            get_metrics(entry, pred)
            return pred
        
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
            messages = build_prompt(entry, dataset_name, base_dir)
            
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
                
                pred = response.choices[0].message.content
                if pred is None:
                    raise ValueError("Empty response from model")
                
                pred = pred.strip()
                
                # Validate prediction format - this will raise InvalidPredictionError if invalid
                # which will trigger a retry
                get_metrics(entry, pred)
                
                return pred
            
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
    ) -> List[str]:
        """
        Evaluate an entry multiple times (sampling).
        
        Args:
            entry: Dataset entry
            dataset_name: Name of the dataset
            num_samples: Number of samples to generate
            base_dir: Base directory for the project
        
        Returns:
            List of predictions
        """
        tasks = [
            self.evaluate_entry(entry, dataset_name, base_dir)
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
    print(f"Evaluating {len(entries)} entries with {num_samples} sample(s) each...")
    
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
    
    # Per-option metrics: aggregate across all samples
    total_correct_options = 0
    total_ground_truth_options = 0
    
    # Per-question-type metrics
    question_type_stats = {}  # question_type -> {total, exactly_correct, first_sample_correct, correct_options, ground_truth_options}
    
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
            detailed_results.append({
                "entry_idx": entry_idx,
                "video": entry.get("video"),
                "question": entry.get("question"),
                "question_type": question_type,
                "error": error,
                "predictions": [],
                "metrics": None,
            })
            # Still track question type stats even for errors (count as total)
            if question_type not in question_type_stats:
                question_type_stats[question_type] = {
                    "total": 0,
                    "exactly_correct": 0,
                    "first_sample_correct": 0,
                    "correct_options": 0,
                    "ground_truth_options": 0,
                }
            question_type_stats[question_type]["total"] += 1
            continue
        
        ground_truth = set(entry["ground_truth"])
        total_ground_truth_options += len(ground_truth) * num_samples
        
        # Initialize question type stats if needed
        if question_type not in question_type_stats:
            question_type_stats[question_type] = {
                "total": 0,
                "exactly_correct": 0,
                "first_sample_correct": 0,
                "correct_options": 0,
                "ground_truth_options": 0,
            }
        question_type_stats[question_type]["total"] += 1
        question_type_stats[question_type]["ground_truth_options"] += len(ground_truth) * num_samples
        
        # Calculate metrics for each prediction
        sample_metrics = []
        question_has_exact_match = False
        question_total_correct = 0
        first_sample_correct = False
        
        for idx, pred in enumerate(predictions):
            try:
                exactly_correct, num_correct = get_metrics(entry, pred)
            except InvalidPredictionError as e:
                # This shouldn't happen since we validate in evaluate_entry,
                # but handle it defensively
                print(f"Warning: Invalid prediction in results for entry {entry.get('video', 'unknown')}: {e}")
                # Treat as incorrect
                exactly_correct, num_correct = 0, 0
            
            sample_metrics.append({
                "prediction": pred,
                "exactly_correct": exactly_correct,
                "num_correct_options": num_correct,
            })
            
            # @1: Check if first sample is correct
            if idx == 0 and exactly_correct:
                first_sample_correct = True
            
            # @k: Check if any sample is correct
            if exactly_correct:
                question_has_exact_match = True
            
            question_total_correct += num_correct
            total_correct_options += num_correct
            question_type_stats[question_type]["correct_options"] += num_correct
        
        if question_has_exact_match:
            question_exactly_correct += 1
            question_type_stats[question_type]["exactly_correct"] += 1
        
        if first_sample_correct:
            question_first_sample_correct += 1
            question_type_stats[question_type]["first_sample_correct"] += 1
        
        detailed_results.append({
            "entry_idx": entry_idx,
            "video": entry.get("video"),
            "question": entry.get("question"),
            "question_type": question_type,
            "ground_truth": entry["ground_truth"],
            "predictions": predictions,
            "sample_metrics": sample_metrics,
            "question_has_exact_match": question_has_exact_match,
            "question_total_correct_options": question_total_correct,
            "first_sample_correct": first_sample_correct,
        })
    
    # Calculate final metrics
    per_question_accuracy = question_exactly_correct / total_questions if total_questions > 0 else 0.0
    per_option_accuracy = total_correct_options / total_ground_truth_options if total_ground_truth_options > 0 else 0.0
    
    # Calculate @1 and @k metrics
    accuracy_at_1 = question_first_sample_correct / total_questions if total_questions > 0 else 0.0
    accuracy_at_k = question_exactly_correct / total_questions if total_questions > 0 else 0.0
    
    metrics_dict = {
        "per_question_accuracy": per_question_accuracy,
        "per_option_accuracy": per_option_accuracy,
        "questions_with_exact_match": question_exactly_correct,
        "total_correct_options": total_correct_options,
        "total_ground_truth_options": total_ground_truth_options,
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
            qtype_metrics = {
                "total_questions": qtype_total,
                "per_question_accuracy": stats["exactly_correct"] / qtype_total,
                "per_option_accuracy": stats["correct_options"] / stats["ground_truth_options"] if stats["ground_truth_options"] > 0 else 0.0,
                "questions_with_exact_match": stats["exactly_correct"],
                "total_correct_options": stats["correct_options"],
                "total_ground_truth_options": stats["ground_truth_options"],
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
        help="Maximum tokens for generation (default: from model config)",
    )
    parser.add_argument(
        "-T", "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: from model config)",
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
        "-p", "--include-predictive",
        action="store_true",
        help="Include predictive.jsonl dataset (default: False, only loads counterfactual_test.jsonl and descriptive.jsonl)",
    )
    parser.add_argument(
        "-C", "--counterfactual-test-size",
        type=int,
        default=None,
        help="Number of entries to load from counterfactual_test.jsonl (default: all)",
    )
    parser.add_argument(
        "-D", "--descriptive-size",
        type=int,
        default=None,
        help="Number of entries to load from descriptive.jsonl (default: all)",
    )
    parser.add_argument(
        "-P", "--predictive-size",
        type=int,
        default=None,
        help="Number of entries to load from predictive.jsonl (default: all, only used if --include-predictive is set)",
    )
    
    args = parser.parse_args()
    
    # Build sizes dict for gather_test_dataset
    # Always include counterfactual_test and descriptive
    sizes = {}
    
    # Check if files exist and get sizes
    dataset_splits_dir = os.path.join(args.base_dir, "datasets", args.dataset, "splits")
    
    # Counterfactual test
    counterfactual_test_path = os.path.join(dataset_splits_dir, "counterfactual_test.jsonl")
    if not os.path.exists(counterfactual_test_path):
        raise FileNotFoundError(f"Dataset file not found: {counterfactual_test_path}")
    if args.counterfactual_test_size is None:
        # Load all entries to determine size
        counterfactual_test_size = len(list(jsonlines.open(counterfactual_test_path)))
    else:
        counterfactual_test_size = args.counterfactual_test_size
    sizes["counterfactual_test"] = counterfactual_test_size
    
    # Descriptive
    descriptive_path = os.path.join(dataset_splits_dir, "descriptive.jsonl")
    if not os.path.exists(descriptive_path):
        raise FileNotFoundError(f"Dataset file not found: {descriptive_path}")
    if args.descriptive_size is None:
        descriptive_size = len(list(jsonlines.open(descriptive_path)))
    else:
        descriptive_size = args.descriptive_size
    sizes["descriptive"] = descriptive_size
    
    # Predictive (optional)
    if args.include_predictive:
        predictive_path = os.path.join(dataset_splits_dir, "predictive.jsonl")
        if not os.path.exists(predictive_path):
            raise FileNotFoundError(f"Dataset file not found: {predictive_path}")
        if args.predictive_size is None:
            predictive_size = len(list(jsonlines.open(predictive_path)))
        else:
            predictive_size = args.predictive_size
        sizes["predictive"] = predictive_size
    
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

