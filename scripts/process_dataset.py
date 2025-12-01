#!/usr/bin/env python3
"""
Process dataset script to split raw QA data into train/test splits.

Filters out bad videos and splits entries based on test video selection,
ensuring counterfactual videos are properly excluded from training.
"""

import argparse
import json
import jsonlines
import os
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Process dataset and create train/test splits"
    )
    parser.add_argument(
        "-d", "--dataset-dir",
        type=str,
        default="datasets",
        help="Dataset directory (default: 'datasets')",
    )
    parser.add_argument(
        "-n", "--dataset-name",
        type=str,
        default="ds2",
        help="Dataset name (e.g., 'ds2')",
    )
    parser.add_argument(
        "-t", "--num-test-videos",
        type=int,
        default=128,
        help="Number of videos to use for test set (default: 128)",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for test video selection (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Build paths
    dataset_path = Path(args.dataset_dir) / args.dataset_name
    raw_qa_path = dataset_path / "raw_qa.jsonl"
    bad_videos_path = dataset_path / "bad_videos.json"
    splits_dir = dataset_path / "splits"
    
    # Validate input files exist
    if not raw_qa_path.exists():
        raise FileNotFoundError(f"Raw QA file not found: {raw_qa_path}")
    if not bad_videos_path.exists():
        raise FileNotFoundError(f"Bad videos file not found: {bad_videos_path}")
    
    # Create splits directory if it doesn't exist
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DATASET PROCESSING")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Number of test videos: {args.num_test_videos}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load raw entries
    print("Loading raw QA data...")
    raw = list(jsonlines.open(raw_qa_path))
    print(f"  Total raw entries: {len(raw)}")
    
    # Load bad videos
    print("Loading bad videos list...")
    bad_videos = set(
        e["video"]
        for e in json.load(open(bad_videos_path))["bad_videos"]
    )
    print(f"  Found {len(bad_videos)} bad videos")
    print()
    
    # Filter out bad videos and group by question type
    print("Filtering entries and grouping by question type...")
    types_to_entries = defaultdict(list)
    all_videos = set()
    excluded_by_bad_video = 0
    
    for entry in raw:
        if entry["video"] in bad_videos:
            excluded_by_bad_video += 1
            continue
        all_videos.add(entry["video"])
        question_type = entry["metadata"]["question_type"]
        types_to_entries[question_type].append(entry)
    
    print(f"  Entries excluded (bad videos): {excluded_by_bad_video}")
    print(f"  Unique videos (after filtering): {len(all_videos)}")
    print(f"  Entries by question type:")
    for qtype in sorted(types_to_entries.keys()):
        count = len(types_to_entries[qtype])
        print(f"    {qtype}: {count}")
    print()
    
    # Select test videos
    print("Selecting test videos...")
    random.seed(args.seed)
    if len(all_videos) < args.num_test_videos:
        print(f"  WARNING: Only {len(all_videos)} videos available, but {args.num_test_videos} requested")
        test_videos = set(all_videos)
    else:
        test_videos = set(random.sample(list(all_videos), args.num_test_videos))
    print(f"  Selected {len(test_videos)} test videos")
    
    # Find counterfactual videos associated with test set
    print("Identifying counterfactual videos for test set...")
    test_counterfactual_videos = set()
    for entry in (
        types_to_entries["counterfactual_velocity"]
        + types_to_entries["counterfactual_position"]
    ):
        if not entry["video"] in test_videos:
            continue
        counterfactual_video = entry["metadata"]["counterfactual_video"]
        test_counterfactual_videos.add(counterfactual_video)
    print(f"  Found {len(test_counterfactual_videos)} counterfactual videos")
    print()
    
    # Split entries into train/test
    print("Splitting entries into train/test sets...")
    types_to_train_entries = defaultdict(list)
    types_to_test_entries = defaultdict(list)
    excluded_entries = 0
    
    for entry in raw:
        video = entry["video"]
        question_type = entry["metadata"]["question_type"]
        counterfactual_video = entry["metadata"].get("counterfactual_video", "dummy")
        
        if video in test_videos:
            types_to_test_entries[question_type].append(entry)
        elif (
            video in test_counterfactual_videos
            or counterfactual_video in test_videos
            or counterfactual_video in test_counterfactual_videos
        ):
            excluded_entries += 1
            # These are videos that we don't want to train on, and don't need to test on either
        else:
            types_to_train_entries[question_type].append(entry)
    
    print(f"  Entries excluded from both train and test: {excluded_entries}")
    print()
    
    # Print summary statistics
    print("=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    
    total_train = sum(len(entries) for entries in types_to_train_entries.values())
    total_test = sum(len(entries) for entries in types_to_test_entries.values())
    
    print(f"Total train entries: {total_train}")
    print(f"Train entries by question type:")
    for qtype in sorted(types_to_train_entries.keys()):
        count = len(types_to_train_entries[qtype])
        print(f"  {qtype}: {count}")
    print()
    
    print(f"Total test entries: {total_test}")
    print(f"Test entries by question type:")
    for qtype in sorted(types_to_test_entries.keys()):
        count = len(types_to_test_entries[qtype])
        print(f"  {qtype}: {count}")
    print()
    
    # Write train splits
    print("Writing train splits...")
    for question_type, entries in sorted(types_to_train_entries.items()):
        output_path = splits_dir / f"train-{question_type}.jsonl"
        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(entries)
        print(f"  Wrote {len(entries)} entries to {output_path}")
    print()
    
    # Write test splits
    print("Writing test splits...")
    for question_type, entries in sorted(types_to_test_entries.items()):
        output_path = splits_dir / f"test-{question_type}.jsonl"
        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(entries)
        print(f"  Wrote {len(entries)} entries to {output_path}")
    print()
    
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
