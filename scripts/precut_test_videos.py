#!/usr/bin/env python3
"""
Pre-cut all test videos in half duration.

This script processes all videos used in test datasets (counterfactual_test, descriptive, predictive)
and creates video_half.mp4 files in the same directory as video.mp4.

This allows us to pre-process videos instead of cutting them during evaluation.
"""

import argparse
import jsonlines
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Set, Tuple
from tqdm import tqdm

# Add parent directory to path to import causal_pool modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from causal_pool.eval.eval_utils import cut_video_first_half


def get_test_video_names(dataset_name: str, base_dir: str = ".") -> Set[str]:
    """
    Get all unique video names from test datasets.
    
    Args:
        dataset_name: Name of the dataset (e.g., '1k_simple')
        base_dir: Base directory for the project
    
    Returns:
        Set of unique video names
    """
    dataset_base_path = os.path.join(base_dir, "datasets", dataset_name)
    splits_dir = os.path.join(dataset_base_path, "splits")
    
    if not os.path.exists(splits_dir):
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    
    video_names = set()
    test_files = ["counterfactual_test.jsonl", "descriptive.jsonl", "predictive.jsonl"]
    
    for filename in test_files:
        filepath = os.path.join(splits_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        print(f"Loading videos from {filename}...")
        with jsonlines.open(filepath) as reader:
            for entry in reader:
                if "video" in entry:
                    video_names.add(entry["video"])
    
    return video_names


def process_single_video(args: Tuple[str, str, bool]) -> Tuple[str, str]:
    """
    Process a single video (worker function for multiprocessing).
    
    Args:
        args: Tuple of (video_name, shots_dir, skip_existing)
            - video_name: Name of the video
            - shots_dir: Directory containing video shots
            - skip_existing: Whether to skip if output already exists
    
    Returns:
        Tuple of (status, video_name) where status is one of:
            - "processed": Successfully cut video
            - "skipped": Skipped (already exists or input missing)
            - "failed": Failed to process
    """
    video_name, shots_dir, skip_existing = args
    
    video_path = os.path.join(shots_dir, video_name, "video.mp4")
    output_path = os.path.join(shots_dir, video_name, "video_half.mp4")
    
    # Check if input video exists
    if not os.path.exists(video_path):
        return ("failed", video_name)
    
    # Check if output already exists
    if skip_existing and os.path.exists(output_path):
        return ("skipped", video_name)
    
    # Cut video
    try:
        cut_video_first_half(video_path, output_path)
        return ("processed", video_name)
    except Exception:
        return ("failed", video_name)


def precut_videos(
    dataset_name: str, 
    base_dir: str = ".", 
    skip_existing: bool = True,
    num_workers: int = 32
) -> None:
    """
    Pre-cut all test videos in half duration using multiprocessing.
    
    Args:
        dataset_name: Name of the dataset (e.g., '1k_simple')
        base_dir: Base directory for the project
        skip_existing: If True, skip videos that already have video_half.mp4
        num_workers: Number of parallel workers (default: 32)
    """
    dataset_base_path = os.path.join(base_dir, "datasets", dataset_name)
    shots_dir = os.path.join(dataset_base_path, "shots")
    
    if not os.path.exists(shots_dir):
        raise FileNotFoundError(f"Shots directory not found: {shots_dir}")
    
    # Get all unique video names from test datasets
    print(f"Collecting video names from test datasets...")
    video_names = get_test_video_names(dataset_name, base_dir)
    print(f"Found {len(video_names)} unique videos in test datasets")
    print(f"Using {num_workers} parallel workers")
    print()
    
    # Prepare arguments for worker function
    video_args = [
        (video_name, shots_dir, skip_existing)
        for video_name in sorted(video_names)
    ]
    
    # Process videos in parallel with progress bar
    processed = 0
    skipped = 0
    failed = 0
    failed_videos = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, video_args),
            total=len(video_args),
            desc="Processing videos",
            unit="video"
        ))
    
    # Count results
    for status, video_name in results:
        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        elif status == "failed":
            failed += 1
            failed_videos.append(video_name)
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total videos: {len(video_names)}")
    print(f"  - Processed: {processed}")
    print(f"  - Skipped (already exists): {skipped}")
    print(f"  - Failed: {failed}")
    
    if failed_videos:
        print()
        print("Failed videos:")
        for video_name in failed_videos[:10]:  # Show first 10
            print(f"  - {video_name}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos) - 10} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cut all test videos in half duration"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., '1k_simple')",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for the project (default: current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-cut videos even if video_half.mp4 already exists",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32)",
    )
    
    args = parser.parse_args()
    
    print(f"Pre-cutting test videos for dataset: {args.dataset}")
    print(f"Base directory: {args.base_dir}")
    print()
    
    try:
        precut_videos(
            args.dataset,
            args.base_dir,
            skip_existing=not args.force,
            num_workers=args.num_workers
        )
        print("✓ All videos processed successfully!")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

