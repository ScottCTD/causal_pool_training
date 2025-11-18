#!/usr/bin/env python3
"""
Override test-predictive.jsonl with questions from all_predictive.jsonl
that use the same test videos as the current test splits.

This script:
1. Extracts the set of test videos from existing test split files
2. Filters all_predictive.jsonl to only include entries with those test videos
3. Writes the filtered entries to test-predictive.jsonl
"""

import argparse
import jsonlines
import os
from pathlib import Path
from collections import defaultdict


def get_test_videos_from_splits(dataset_name: str, dataset_dir: str = "datasets") -> set:
    """
    Extract the set of test videos from existing test split files.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'ds1')
        dataset_dir: Directory containing datasets (default: 'datasets')
    
    Returns:
        Set of test video names
    """
    dataset_path = Path(dataset_dir) / dataset_name
    splits_dir = dataset_path / "splits"
    
    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    
    test_files = [
        "test-counterfactual_velocity.jsonl",
        "test-counterfactual_position.jsonl",
        "test-descriptive.jsonl",
        "test-predictive.jsonl",
    ]
    
    test_videos = set()
    
    for filename in test_files:
        filepath = splits_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        print(f"Loading test videos from {filename}...")
        with jsonlines.open(filepath) as reader:
            for entry in reader:
                if "video" in entry:
                    test_videos.add(entry["video"])
    
    return test_videos


def filter_predictive_entries(
    all_predictive_path: Path,
    test_videos: set,
) -> list:
    """
    Filter all_predictive.jsonl to only include entries with test videos.
    
    Args:
        all_predictive_path: Path to all_predictive.jsonl
        test_videos: Set of test video names
    
    Returns:
        List of filtered entries
    """
    if not all_predictive_path.exists():
        raise FileNotFoundError(f"all_predictive.jsonl not found: {all_predictive_path}")
    
    print(f"Loading entries from {all_predictive_path}...")
    all_entries = list(jsonlines.open(all_predictive_path))
    print(f"  Total entries: {len(all_entries)}")
    
    filtered_entries = []
    for entry in all_entries:
        if entry.get("video") in test_videos:
            filtered_entries.append(entry)
    
    print(f"  Filtered entries (test videos only): {len(filtered_entries)}")
    
    return filtered_entries


def main():
    parser = argparse.ArgumentParser(
        description="Override test-predictive.jsonl with filtered entries from all_predictive.jsonl"
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
        required=True,
        help="Dataset name (e.g., 'ds1')",
    )
    
    args = parser.parse_args()
    
    # Build paths
    dataset_path = Path(args.dataset_dir) / args.dataset_name
    all_predictive_path = dataset_path / "all_predictive.jsonl"
    splits_dir = dataset_path / "splits"
    test_predictive_path = splits_dir / "test-predictive.jsonl"
    
    print("=" * 60)
    print("OVERRIDE TEST-PREDICTIVE.JSONL")
    print("=" * 60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Dataset directory: {args.dataset_dir}")
    print()
    
    # Step 1: Extract test videos from existing test splits
    print("Step 1: Extracting test videos from existing test splits...")
    test_videos = get_test_videos_from_splits(args.dataset_name, args.dataset_dir)
    print(f"  Found {len(test_videos)} unique test videos")
    print()
    
    # Step 2: Filter all_predictive.jsonl
    print("Step 2: Filtering all_predictive.jsonl...")
    filtered_entries = filter_predictive_entries(all_predictive_path, test_videos)
    print()
    
    # Step 3: Write to test-predictive.jsonl
    print("Step 3: Writing filtered entries to test-predictive.jsonl...")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(test_predictive_path, "w") as writer:
        writer.write_all(filtered_entries)
    
    print(f"  Wrote {len(filtered_entries)} entries to {test_predictive_path}")
    
    # Print summary statistics
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    # Count entries per video
    video_counts = defaultdict(int)
    for entry in filtered_entries:
        video_counts[entry["video"]] += 1
    
    print(f"Total entries: {len(filtered_entries)}")
    print(f"Unique videos: {len(video_counts)}")
    print(f"Average entries per video: {len(filtered_entries) / len(video_counts):.2f}")
    print()
    
    # Show top videos by entry count
    print("Top 10 videos by entry count:")
    sorted_videos = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)
    for video, count in sorted_videos[:10]:
        print(f"  {video}: {count} entries")
    
    print()
    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

