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
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
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


class InvalidPredictionError(ValueError):
    """Raised when prediction format is invalid (not pure A-Z or has duplicates)."""
    pass


# Create retry condition by combining existing helpers
# Retry on InvalidPredictionError OR API errors OR (everything except BadRequestError and ValueError)
# Note: InvalidPredictionError is explicitly included even though it's a ValueError subclass
_retry_condition = (
    retry_if_exception_type((InvalidPredictionError, APIError, APIConnectionError, APITimeoutError))
    | retry_if_not_exception_type((BadRequestError, ValueError))
)


def normalize_model_name(model_name: str) -> str:
    """Normalize model name for file naming."""
    return model_name.split("/")[-1].replace("-", "_").lower()


def get_metrics(entry: Dict[str, Any], pred: str) -> Tuple[int, int]:
    """
    Returns (exactly correct or not, how many options were correct).
    
    Args:
        entry: Dataset entry with 'ground_truth' field
        pred: Prediction string (e.g., "AC")
    
    Returns:
        Tuple of (exactly_correct, num_correct_options)
    
    Raises:
        InvalidPredictionError: If prediction format is invalid (not pure A-Z or has duplicates)
    """
    pred = pred.strip()
    
    if not all(c.isalpha() and c.isupper() for c in pred):
        raise InvalidPredictionError(f"Prediction contains non-A-Z characters: {pred!r}")
    
    selected_options = set(ord(c) - ord("A") for c in pred)

    if len(selected_options) != len(pred):  # duplicate options
        raise InvalidPredictionError(f"Prediction contains duplicate options: {pred!r}")
    
    ground_truth = set(entry["ground_truth"])
    
    exactly_correct = int(selected_options == ground_truth)
    num_correct = len(selected_options & ground_truth)
    
    return exactly_correct, num_correct


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Duration in seconds
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to get video duration for {video_path}: {e}")


def cut_video_first_half(video_path: str, output_path: str) -> None:
    """
    Cut video to first half using ffmpeg.
    
    First tries codec copy (fast), falls back to re-encoding if that fails.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save cut video
    """
    duration = get_video_duration(video_path)
    half_duration = duration / 2.0
    
    # First try codec copy (fast, no re-encoding)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-t", str(half_duration),
                "-c", "copy",  # Copy codec to avoid re-encoding (faster)
                "-y",  # Overwrite output file
                output_path
            ],
            capture_output=True,
            check=True,
        )
        return  # Success with codec copy
    except subprocess.CalledProcessError:
        # Codec copy failed (e.g., cutting at non-keyframe), fall through to re-encoding
        pass
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to use predictive question type.")
    
    # Fall back to re-encoding if codec copy fails (e.g., cutting at non-keyframe)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-t", str(half_duration),
                "-c:v", "libx264",  # Re-encode video
                "-c:a", "aac",  # Re-encode audio
                "-preset", "fast",  # Faster encoding
                "-y",  # Overwrite output file
                output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Failed to cut video {video_path}: {error_msg}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to use predictive question type.")


def build_prompt(entry: Dict[str, Any], dataset_name: str, base_dir: str = ".") -> List[Dict[str, Any]]:
    """
    Build prompt for a dataset entry.
    
    For predictive questions, only the first half of the video is used.
    
    Args:
        entry: Dataset entry
        dataset_name: Name of the dataset
        base_dir: Base directory for the project
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    video_path = os.path.join(
        base_dir, "datasets", dataset_name, "shots", entry["video"], f"video_{entry['video']}.mp4"
    )
    
    question_prompt = f"{entry['question']}\n"
    for i, choice in enumerate(entry["options"]):
        question_prompt += f"{chr(ord('A') + i)}. {choice}\n"
    
    question_prompt += "\nPlease select the correct option(s). Don't write anything else than the option letter(s). Example: AC."
    
    # Check if this is a predictive question type
    question_type = entry.get("metadata", {}).get("question_type", "")
    is_predictive = question_type == "predictive"
    
    # For predictive questions, cut video to first half
    video_to_encode = video_path
    temp_video_path = None
    
    if is_predictive:
        # Create temporary file for cut video
        temp_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)
        try:
            cut_video_first_half(video_path, temp_video_path)
            video_to_encode = temp_video_path
        except Exception as e:
            # Clean up temp file if cutting fails
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            raise RuntimeError(f"Failed to process predictive video: {e}")
    
    # Read and encode video
    try:
        with open(video_to_encode, "rb") as video_file:
            video_b64 = base64.b64encode(video_file.read()).decode("utf-8")
    finally:
        # Clean up temporary cut video if it was created
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
    
    return [{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            },
            {"type": "text", "text": question_prompt},
        ],
    }]


class AsyncEvaluator:
    """Async evaluator with retry logic."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_tokens: int = 20,
        temperature: float = 0.8,
        extra_body: Optional[Dict[str, Any]] = None,
        max_retries: int = 10,
    ):
        """
        Initialize async evaluator.
        
        Args:
            base_url: Base URL for the API
            model: Model name
            api_key: API key (default: "EMPTY")
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            extra_body: Extra body parameters for API
            max_retries: Maximum number of retries
        """
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = extra_body or {
            "top_k": 20,
            "top_p": 0.8,
            "repetition_penalty": 1.0,
            "presence_penalty": 1.5,
        }
        self.max_retries = max_retries
        # Track which entries have had their first error printed
        self._first_error_printed = set()
    
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
                response = await self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_body=self.extra_body,
                )
                
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
    dataset_path: str,
    evaluator: AsyncEvaluator,
    num_samples: int = 1,
    max_concurrent: int = 10,
    base_dir: str = ".",
    max_entries: Optional[int] = None,
    exclude_predictive: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate entire dataset asynchronously.
    
    Args:
        dataset_path: Path to JSONL dataset file
        evaluator: AsyncEvaluator instance
        num_samples: Number of samples per question
        max_concurrent: Maximum concurrent requests
        base_dir: Base directory for the project
    
    Returns:
        Dictionary with evaluation results and metrics
    """
    # Extract dataset name from path
    dataset_name = Path(dataset_path).parent.name
    
    # Load dataset
    entries = list(jsonlines.open(dataset_path))
    if max_entries is not None:
        entries = entries[:max_entries]
        print(f"Limited to {len(entries)} entries (max_entries={max_entries})")

    # Optionally exclude predictive question type
    if exclude_predictive:
        before_count = len(entries)
        entries = [
            e
            for e in entries
            if e.get("metadata", {}).get("question_type", "") != "predictive"
        ]
        excluded = before_count - len(entries)
        print(f"Excluded {excluded} predictive question(s); {len(entries)} remaining.")

    print(f"Loaded {len(entries)} entries from {dataset_path}")
    
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
    
    # For high concurrency, process in smaller batches to ensure event loop responsiveness
    # Smaller batches = more frequent opportunities for Ctrl+C to work
    if max_concurrent > 100:
        batch_size = 50  # Small batches for very high concurrency
    elif max_concurrent > 50:
        batch_size = max_concurrent
    else:
        batch_size = len(entries)  # Single batch for low concurrency
    
    results = []
    all_tasks = []
    
    with tqdm(total=len(entries), desc="Evaluating") as pbar:
        try:
            for batch_start in range(0, len(entries), batch_size):
                batch_end = min(batch_start + batch_size, len(entries))
                batch_entries = entries[batch_start:batch_end]
                
                # Create tasks for this batch
                tasks = [
                    asyncio.create_task(evaluate_with_idx(batch_start + idx, entry))
                    for idx, entry in enumerate(batch_entries)
                ]
                all_tasks.extend(tasks)
                
                # Wait for batch to complete with gather (allows proper cancellation)
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and handle exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        # This shouldn't happen due to exception handling in evaluate_with_semaphore
                        # but handle it just in case
                        print(f"Unexpected exception in batch: {result}")
                    else:
                        results.append(result)
                    pbar.update(1)
                
                # Yield control to event loop briefly between batches
                # This ensures Ctrl+C can be processed
                await asyncio.sleep(0)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted! Cancelling remaining tasks...")
            # Cancel all pending tasks
            for task in all_tasks:
                if not task.done():
                    task.cancel()
            # Wait for cancellations to complete
            await asyncio.gather(*all_tasks, return_exceptions=True)
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
        "dataset_path": dataset_path,
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
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., '1k_simple')",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://trig0002:8000/v1",
        help="Base URL for the API",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'Qwen/Qwen3-VL-4B-Instruct')",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key (default: 'EMPTY')",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per question (default: 1)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10). High values (128, 512) are now supported with automatic batching.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum tokens for generation (default: 20)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for the project (default: current directory)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to evaluate (for testing, default: all)",
    )
    parser.add_argument(
        "--exclude-predictive",
        action="store_true",
        help="Exclude questions with metadata.question_type == 'predictive'",
    )
    
    args = parser.parse_args()
    
    # Build dataset path
    dataset_path = os.path.join(args.base_dir, "datasets", args.dataset, f"{args.dataset}.jsonl")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
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
    print(f"  Max concurrent: {args.max_concurrent}")
    
    # Warn about very high concurrency
    if args.max_concurrent > 100:
        print(f"\n⚠️  WARNING: Very high concurrency ({args.max_concurrent}) may cause:")
        print("   - Memory issues")
        print("   - Connection pool exhaustion")
        print("   Consider using a lower value (e.g., 50-100) for better stability.")
        print("   Note: Processing will use smaller batches (50) to maintain responsiveness.\n")
    
    try:
        results = await evaluate_dataset(
            dataset_path=dataset_path,
            evaluator=evaluator,
            num_samples=args.num_samples,
            max_concurrent=args.max_concurrent,
            base_dir=args.base_dir,
            max_entries=args.max_entries,
            exclude_predictive=args.exclude_predictive,
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
            "datasets",
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

