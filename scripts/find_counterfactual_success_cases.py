#!/usr/bin/env python3
"""
Find examples where counterfactual questions are answered correctly
but descriptive/predictive questions are answered incorrectly.

Outputs results to a JSON file keyed by shot_id.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

import jsonlines


def load_results(results_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def is_counterfactual(question_type: str) -> bool:
    """Check if question type is counterfactual."""
    return question_type in ["counterfactual_velocity", "counterfactual_position"]


def is_descriptive_or_predictive(question_type: str) -> bool:
    """Check if question type is descriptive or predictive."""
    return question_type in ["descriptive", "predictive"]


def load_counterfactual_video_mapping(dataset_name: str, base_dir: str = ".") -> Dict[tuple, str]:
    """
    Load counterfactual_video mapping from dataset files.
    
    Creates a mapping from (video, question_type, ground_truth_tuple) to counterfactual_video.
    For counterfactual questions, the counterfactual_video is the shot ID of the
    counterfactual scenario.
    
    Uses ground_truth to uniquely identify each counterfactual question since
    multiple counterfactual questions can exist for the same video.
    
    Returns:
        Dictionary mapping (video, question_type, ground_truth_tuple) -> counterfactual_video
    """
    mapping = {}
    dataset_splits_dir = os.path.join(base_dir, "datasets", dataset_name, "splits")
    
    # Load counterfactual velocity entries
    cf_velocity_path = os.path.join(dataset_splits_dir, "test-counterfactual_velocity.jsonl")
    if os.path.exists(cf_velocity_path):
        with jsonlines.open(cf_velocity_path) as reader:
            for entry in reader:
                video = entry.get("video")
                counterfactual_video = entry.get("metadata", {}).get("counterfactual_video")
                ground_truth = entry.get("ground_truth")
                if video and counterfactual_video and ground_truth is not None:
                    # Convert ground_truth to tuple for hashing
                    gt_tuple = tuple(sorted(ground_truth)) if isinstance(ground_truth, (list, set)) else (ground_truth,)
                    mapping[(video, "counterfactual_velocity", gt_tuple)] = counterfactual_video
    
    # Load counterfactual position entries
    cf_position_path = os.path.join(dataset_splits_dir, "test-counterfactual_position.jsonl")
    if os.path.exists(cf_position_path):
        with jsonlines.open(cf_position_path) as reader:
            for entry in reader:
                video = entry.get("video")
                counterfactual_video = entry.get("metadata", {}).get("counterfactual_video")
                ground_truth = entry.get("ground_truth")
                if video and counterfactual_video and ground_truth is not None:
                    # Convert ground_truth to tuple for hashing
                    gt_tuple = tuple(sorted(ground_truth)) if isinstance(ground_truth, (list, set)) else (ground_truth,)
                    mapping[(video, "counterfactual_position", gt_tuple)] = counterfactual_video
    
    return mapping


def find_counterfactual_success_cases(results: Dict[str, Any], cf_video_mapping: Dict[tuple, str]) -> Dict[str, Dict[str, Any]]:
    """
    Find shot_ids where:
    - At least one counterfactual question is correct
    - At least one descriptive/predictive question is incorrect
    
    Returns a dictionary keyed by shot_id with all relevant questions.
    """
    detailed_results = results.get("detailed_results", [])
    
    # Group results by shot_id (video)
    by_shot = defaultdict(list)
    for entry in detailed_results:
        shot_id = entry.get("video")
        if shot_id:
            by_shot[shot_id].append(entry)
    
    # Find shots that match our criteria
    matching_shots = {}
    
    for shot_id, entries in by_shot.items():
        # Check if ALL counterfactual questions are correct
        counterfactual_entries = [e for e in entries if is_counterfactual(e.get("question_type", ""))]
        all_counterfactual_correct = True
        if len(counterfactual_entries) == 0:
            # Skip shots with no counterfactual questions
            continue
        for entry in counterfactual_entries:
            if not entry.get("question_has_exact_match", False):
                all_counterfactual_correct = False
                break
        
        # Check if there's at least one incorrect descriptive/predictive question
        has_incorrect_descriptive_predictive = False
        for entry in entries:
            if is_descriptive_or_predictive(entry.get("question_type", "")):
                if not entry.get("question_has_exact_match", True):
                    has_incorrect_descriptive_predictive = True
                    break
        
        # If both conditions are met, include this shot
        if all_counterfactual_correct and has_incorrect_descriptive_predictive:
            # Store all questions for this shot, organized by type
            shot_data = {
                "shot_id": shot_id,
                "counterfactual_questions": [],
                "descriptive_questions": [],
                "predictive_questions": [],
            }
            
            for entry in entries:
                question_type = entry.get("question_type", "")
                question_data = {
                    "question_type": question_type,
                    "entry_idx": entry.get("entry_idx"),
                    "correct": entry.get("question_has_exact_match", False),
                    "ground_truth": entry.get("ground_truth"),
                    "predictions": entry.get("predictions", []),
                    "question_fraction_correct": entry.get("question_fraction_correct", 0.0),
                    # Extract question text from prompt
                    "question": None,
                    # Counterfactual shot ID (only for counterfactual questions)
                    "counterfactual_shot_id": None,
                }
                
                # Get counterfactual shot ID for counterfactual questions
                if is_counterfactual(question_type):
                    video = entry.get("video")
                    ground_truth = entry.get("ground_truth")
                    if video and ground_truth is not None:
                        # Convert ground_truth to tuple for matching
                        gt_tuple = tuple(sorted(ground_truth)) if isinstance(ground_truth, (list, set)) else (ground_truth,)
                        cf_video = cf_video_mapping.get((video, question_type, gt_tuple))
                        if cf_video:
                            question_data["counterfactual_shot_id"] = cf_video
                
                # Extract full question text with options from prompt if available
                prompts = entry.get("prompts", [])
                if prompts and len(prompts) > 0:
                    first_prompt = prompts[0]
                    if isinstance(first_prompt, list) and len(first_prompt) > 0:
                        content = first_prompt[0].get("content", [])
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text = item.get("text", "")
                                # Extract question with all options (from "Question:" to "Please select")
                                if "Question:" in text:
                                    question_start = text.find("Question:")
                                    # Find the end marker ("Please select" or end of text)
                                    end_marker = text.find("\nPlease select", question_start)
                                    if end_marker == -1:
                                        end_marker = text.find("\n\nPlease select", question_start)
                                    if end_marker != -1:
                                        # Extract from "Question:" to "Please select"
                                        question_data["question"] = text[question_start:end_marker].strip()
                                    else:
                                        # Fallback: take everything from "Question:" to the end
                                        question_data["question"] = text[question_start:].strip()
                                break
                
                if is_counterfactual(question_type):
                    shot_data["counterfactual_questions"].append(question_data)
                elif question_type == "descriptive":
                    shot_data["descriptive_questions"].append(question_data)
                elif question_type == "predictive":
                    shot_data["predictive_questions"].append(question_data)
            
            matching_shots[shot_id] = shot_data
    
    return matching_shots


def main():
    """Main function."""
    # Path to results file
    results_path = Path(__file__).parent.parent / "results" / "ds2" / "eval_causalpool_4b_cf.json"
    
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        sys.exit(1)
    
    print(f"Loading results from {results_path}...")
    results = load_results(str(results_path))
    
    # Load counterfactual video mapping
    dataset_name = results.get("dataset", "ds2")
    base_dir = Path(__file__).parent.parent
    print(f"Loading counterfactual video mapping for dataset {dataset_name}...")
    cf_video_mapping = load_counterfactual_video_mapping(dataset_name, str(base_dir))
    print(f"Loaded {len(cf_video_mapping)} counterfactual video mappings")
    
    print("Finding matching examples...")
    matching_shots = find_counterfactual_success_cases(results, cf_video_mapping)
    
    print(f"Found {len(matching_shots)} shot(s) matching the criteria")
    
    # Output to JSON file
    output_path = Path(__file__).parent.parent / "results" / "ds2" / "counterfactual_success_cases.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(matching_shots, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Print summary
    print("\nSummary:")
    for shot_id, shot_data in matching_shots.items():
        cf_correct = sum(1 for q in shot_data["counterfactual_questions"] if q["correct"])
        cf_total = len(shot_data["counterfactual_questions"])
        desc_incorrect = sum(1 for q in shot_data["descriptive_questions"] if not q["correct"])
        desc_total = len(shot_data["descriptive_questions"])
        pred_incorrect = sum(1 for q in shot_data["predictive_questions"] if not q["correct"])
        pred_total = len(shot_data["predictive_questions"])
        
        print(f"\n  {shot_id}:")
        print(f"    Counterfactual: {cf_correct}/{cf_total} correct")
        print(f"    Descriptive: {desc_incorrect}/{desc_total} incorrect")
        print(f"    Predictive: {pred_incorrect}/{pred_total} incorrect")


if __name__ == "__main__":
    main()

