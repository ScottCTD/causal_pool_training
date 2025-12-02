#!/usr/bin/env python3
"""
Recompute fractional correctness metrics for existing evaluation results.

Updates all result files to use the new fractional correctness metric:
- question_fraction_correct = num_correct_cameras / total_cameras
- per_question_accuracy = average(question_fraction_correct) across all questions
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def recompute_question_metrics(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recompute fractional correctness metrics for a single question entry.
    
    Args:
        entry: Detailed result entry with predictions and sample_metrics
        
    Returns:
        Updated entry with fractional correctness metrics
    """
    # Get predictions and sample metrics
    predictions = entry.get("predictions", [])
    sample_metrics = entry.get("sample_metrics", [])
    
    # Count correct camera angles
    num_correct_cameras = 0
    total_cameras = len(predictions)
    
    # Count correct predictions from sample_metrics
    for metric in sample_metrics:
        if metric.get("exactly_correct", 0) == 1:
            num_correct_cameras += 1
    
    # Calculate fractional correctness
    question_fraction_correct = num_correct_cameras / total_cameras if total_cameras > 0 else 0.0
    
    # Update entry with new fields
    entry["question_fraction_correct"] = question_fraction_correct
    entry["num_correct_cameras"] = num_correct_cameras
    entry["total_cameras"] = total_cameras
    
    # Keep question_has_exact_match for backward compatibility (any camera correct)
    entry["question_has_exact_match"] = num_correct_cameras > 0
    
    return entry


def recompute_aggregated_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recompute aggregated metrics from detailed results.
    
    Args:
        results: Full results dictionary
        
    Returns:
        Updated results dictionary with recomputed metrics
    """
    detailed_results = results.get("detailed_results", [])
    total_questions = len(detailed_results)
    
    if total_questions == 0:
        return results
    
    # Accumulate fractional correctness
    question_fraction_sum = 0.0
    question_first_sample_correct = 0
    
    # Per-question-type stats
    question_type_stats = {}
    
    # Per-option accuracy (already computed, just accumulate)
    total_per_option_acc = 0.0
    total_samples_for_per_option = 0
    
    # Token usage (already computed)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    samples_with_token_usage = 0
    
    for entry in detailed_results:
        # Skip errored entries
        if entry.get("error"):
            question_type = entry.get("question_type", "unknown")
            if question_type not in question_type_stats:
                question_type_stats[question_type] = {
                    "total": 0,
                    "exactly_correct_sum": 0.0,
                    "first_sample_correct": 0,
                    "per_option_acc_sum": 0.0,
                    "per_option_samples": 0,
                }
            question_type_stats[question_type]["total"] += 1
            continue
        
        # Recompute question-level metrics if not already present
        if "question_fraction_correct" not in entry:
            entry = recompute_question_metrics(entry)
        
        question_fraction = entry.get("question_fraction_correct", 0.0)
        question_type = entry.get("question_type", "unknown")
        
        # Accumulate fractional correctness
        question_fraction_sum += question_fraction
        
        # Track first sample correctness
        if entry.get("first_sample_correct", False):
            question_first_sample_correct += 1
        
        # Initialize question type stats
        if question_type not in question_type_stats:
            question_type_stats[question_type] = {
                "total": 0,
                "exactly_correct_sum": 0.0,
                "first_sample_correct": 0,
                "per_option_acc_sum": 0.0,
                "per_option_samples": 0,
            }
        
        question_type_stats[question_type]["total"] += 1
        question_type_stats[question_type]["exactly_correct_sum"] += question_fraction
        
        if entry.get("first_sample_correct", False):
            question_type_stats[question_type]["first_sample_correct"] += 1
        
        # Accumulate per-option accuracy from sample_metrics
        sample_metrics = entry.get("sample_metrics", [])
        for metric in sample_metrics:
            per_option_acc = metric.get("per_option_accuracy", 0.0)
            total_per_option_acc += per_option_acc
            total_samples_for_per_option += 1
            question_type_stats[question_type]["per_option_acc_sum"] += per_option_acc
            question_type_stats[question_type]["per_option_samples"] += 1
        
        # Accumulate token usage
        token_usage_list = entry.get("token_usage", [])
        for token_usage in token_usage_list:
            if token_usage is not None:
                if token_usage.get('prompt_tokens') is not None:
                    total_prompt_tokens += token_usage['prompt_tokens']
                if token_usage.get('completion_tokens') is not None:
                    total_completion_tokens += token_usage['completion_tokens']
                if token_usage.get('total_tokens') is not None:
                    total_tokens += token_usage['total_tokens']
                samples_with_token_usage += 1
    
    # Calculate final metrics
    per_question_accuracy = question_fraction_sum / total_questions if total_questions > 0 else 0.0
    per_option_accuracy = total_per_option_acc / total_samples_for_per_option if total_samples_for_per_option > 0 else 0.0
    
    # Build metrics dictionary
    metrics_dict = {
        "per_question_accuracy": per_question_accuracy,
        "per_option_accuracy": per_option_accuracy,
        "total_samples": total_samples_for_per_option,
        "token_usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "samples_with_token_usage": samples_with_token_usage,
        },
    }
    
    # Add @1 metrics if num_samples > 1
    num_samples = results.get("num_samples", 1)
    if num_samples > 1:
        accuracy_at_1 = question_first_sample_correct / total_questions if total_questions > 0 else 0.0
        metrics_dict["accuracy@1"] = accuracy_at_1
        metrics_dict[f"accuracy@{num_samples}"] = per_question_accuracy
        metrics_dict["questions_with_first_sample_correct"] = question_first_sample_correct
    
    # Calculate per-question-type metrics
    per_question_type_metrics = {}
    for qtype, stats in question_type_stats.items():
        qtype_total = stats["total"]
        if qtype_total > 0:
            qtype_per_option_acc = stats["per_option_acc_sum"] / stats["per_option_samples"] if stats["per_option_samples"] > 0 else 0.0
            qtype_per_question_acc = stats["exactly_correct_sum"] / qtype_total
            
            qtype_metrics = {
                "total_questions": qtype_total,
                "per_question_accuracy": qtype_per_question_acc,
                "per_option_accuracy": qtype_per_option_acc,
                "total_samples": stats["per_option_samples"],
            }
            
            if num_samples > 1:
                qtype_metrics["accuracy@1"] = stats["first_sample_correct"] / qtype_total
                qtype_metrics[f"accuracy@{num_samples}"] = qtype_per_question_acc
                qtype_metrics["questions_with_first_sample_correct"] = stats["first_sample_correct"]
            
            per_question_type_metrics[qtype] = qtype_metrics
    
    metrics_dict["per_question_type"] = per_question_type_metrics
    
    # Update results
    results["metrics"] = metrics_dict
    results["detailed_results"] = detailed_results
    
    return results


def process_result_file(filepath: Path) -> bool:
    """
    Process a single result file to recompute metrics.
    
    Args:
        filepath: Path to result JSON file
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Processing {filepath.name}...")
    
    try:
        # Load results
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Check if already updated (has question_fraction_correct in first entry)
        if results.get("detailed_results"):
            first_entry = results["detailed_results"][0]
            if "question_fraction_correct" in first_entry:
                print(f"  Already updated, skipping...")
                return True
        
        # Recompute metrics
        results = recompute_aggregated_metrics(results)
        
        # Save back
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Updated: per_question_accuracy = {results['metrics']['per_question_accuracy']:.4f}")
        return True
        
    except Exception as e:
        print(f"  Error processing {filepath.name}: {e}")
        return False


def main():
    """Main function to process all ds2 result files."""
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results" / "ds2"
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Find all JSON result files
    result_files = list(results_dir.glob("eval_*.json"))
    
    # Also check subdirectories
    fps_dir = results_dir / "eval_fps"
    if fps_dir.exists():
        result_files.extend(fps_dir.glob("eval_*.json"))
    
    if not result_files:
        print("No result files found!")
        sys.exit(1)
    
    print(f"Found {len(result_files)} result file(s) to process\n")
    
    success_count = 0
    for filepath in sorted(result_files):
        if process_result_file(filepath):
            success_count += 1
        print()
    
    print(f"Successfully updated {success_count}/{len(result_files)} file(s)")


if __name__ == "__main__":
    main()

