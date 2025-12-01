#!/usr/bin/env python3
"""
Pre-cut all test videos to a specified fraction of their duration.

This script processes all videos used in test datasets:
- test-counterfactual_velocity.jsonl
- test-counterfactual_position.jsonl
- test-descriptive.jsonl
- test-predictive.jsonl

For each shot directory containing multiple camera angle videos:
1. Renames all existing .mp4 files to *-full.mp4
2. Creates *-partial.mp4 files by cutting each *-full.mp4

This allows us to pre-process videos instead of cutting them during evaluation.
"""

import argparse
import jsonlines
import multiprocessing
import os
import sys
from pathlib import Path
from typing import List, Set, Tuple
from tqdm import tqdm

# Add parent directory to path to import causal_pool modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from causal_pool.data.video_utils import cut_video_fraction


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
    test_files = [
        "test-counterfactual_velocity.jsonl",
        "test-counterfactual_position.jsonl",
        "test-descriptive.jsonl",
        "test-predictive.jsonl",
    ]
    
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


def process_single_shot(args: Tuple[str, str, bool, float]) -> Tuple[str, int, int, int, List[str]]:
    """
    Process a single shot directory (worker function for multiprocessing).
    
    For each shot directory:
    1. Finds all .mp4 files that don't end with -full.mp4 or -partial.mp4
    2. Renames them to *-full.mp4
    3. Creates *-partial.mp4 for each *-full.mp4
    
    Args:
        args: Tuple of (video_name, shots_dir, skip_existing, fraction)
            - video_name: Name of the video/shot directory
            - shots_dir: Directory containing video shots
            - skip_existing: Whether to skip if output already exists
            - fraction: Fraction of video to retain (e.g., 0.3 for first 30%)
    
    Returns:
        Tuple of (status, renamed_count, processed_count, failed_count, failed_files)
            - status: "processed", "skipped", or "failed"
            - renamed_count: Number of videos renamed to *-full.mp4
            - processed_count: Number of partial videos created
            - failed_count: Number of failed operations
            - failed_files: List of filenames that failed
    """
    video_name, shots_dir, skip_existing, fraction = args
    
    shot_dir = os.path.join(shots_dir, video_name)
    
    if not os.path.exists(shot_dir):
        return ("failed", 0, 0, 1, [video_name])
    
    renamed_count = 0
    processed_count = 0
    failed_count = 0
    failed_files = []
    
    # Find all .mp4 files in the shot directory
    try:
        all_files = os.listdir(shot_dir)
        mp4_files = [f for f in all_files if f.endswith('.mp4')]
    except Exception:
        return ("failed", 0, 0, 1, [video_name])
    
    # Step 1: Rename existing videos to *-full.mp4
    for filename in mp4_files:
        # Skip files that already end with -full.mp4 or -partial.mp4
        if filename.endswith('-full.mp4') or filename.endswith('-partial.mp4'):
            continue
        
        # Rename to *-full.mp4
        base_name = os.path.splitext(filename)[0]  # Remove .mp4 extension
        new_filename = f"{base_name}-full.mp4"
        old_path = os.path.join(shot_dir, filename)
        new_path = os.path.join(shot_dir, new_filename)
        
        # Skip if already renamed (shouldn't happen, but be safe)
        if os.path.exists(new_path):
            continue
        
        try:
            os.rename(old_path, new_path)
            renamed_count += 1
        except Exception:
            failed_count += 1
            failed_files.append(f"{video_name}/{filename}")
    
    # Step 2: Create *-partial.mp4 for each *-full.mp4
    try:
        all_files_after_rename = os.listdir(shot_dir)
        full_videos = [f for f in all_files_after_rename if f.endswith('-full.mp4')]
    except Exception:
        return ("failed", renamed_count, processed_count, failed_count + 1, failed_files)
    
    for full_filename in full_videos:
        # Create corresponding partial filename
        base_name = full_filename[:-9]  # Remove '-full.mp4' suffix
        partial_filename = f"{base_name}-partial.mp4"
        full_path = os.path.join(shot_dir, full_filename)
        partial_path = os.path.join(shot_dir, partial_filename)
        
        # Check if input exists
        if not os.path.exists(full_path):
            failed_count += 1
            failed_files.append(f"{video_name}/{full_filename}")
            continue
        
        # Check if output already exists
        if skip_existing and os.path.exists(partial_path):
            continue
        
        # If not skipping existing files (i.e., --force), delete existing output file
        if not skip_existing and os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except Exception:
                # If deletion fails, still try to cut (ffmpeg -y should overwrite)
                pass
        
        # Cut video
        try:
            cut_video_fraction(full_path, partial_path, fraction=fraction)
            processed_count += 1
        except Exception:
            failed_count += 1
            failed_files.append(f"{video_name}/{full_filename}")
    
    # Determine overall status
    if failed_count > 0:
        status = "failed"
    elif processed_count == 0 and renamed_count == 0:
        status = "skipped"
    else:
        status = "processed"
    
    return (status, renamed_count, processed_count, failed_count, failed_files)


def precut_videos(
    dataset_name: str, 
    base_dir: str = ".", 
    skip_existing: bool = True,
    num_workers: int = 32,
    fraction: float = 0.5
) -> None:
    """
    Pre-cut all test videos to a specified fraction of their duration using multiprocessing.
    
    Args:
        dataset_name: Name of the dataset (e.g., '1k_simple')
        base_dir: Base directory for the project
        skip_existing: If True, skip videos that already have the output file
        num_workers: Number of parallel workers (default: 32)
        fraction: Fraction of video to retain (e.g., 0.3 for first 30%, default: 0.5)
    """
    if not 0 < fraction <= 1.0:
        raise ValueError(f"Fraction must be between 0 and 1, got {fraction}")
    
    dataset_base_path = os.path.join(base_dir, "datasets", dataset_name)
    shots_dir = os.path.join(dataset_base_path, "shots")
    
    if not os.path.exists(shots_dir):
        raise FileNotFoundError(f"Shots directory not found: {shots_dir}")
    
    # Get all unique video names from test datasets
    print(f"Collecting video names from test datasets...")
    video_names = get_test_video_names(dataset_name, base_dir)
    print(f"Found {len(video_names)} unique videos in test datasets")
    print(f"Using {num_workers} parallel workers")
    print(f"Cutting videos to first {fraction*100:.1f}% of duration")
    print()
    
    # Prepare arguments for worker function
    video_args = [
        (video_name, shots_dir, skip_existing, fraction)
        for video_name in sorted(video_names)
    ]
    
    # Process shot directories in parallel with progress bar
    processed_shots = 0
    skipped_shots = 0
    failed_shots = 0
    total_renamed = 0
    total_processed = 0
    total_failed = 0
    failed_files = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_shot, video_args),
            total=len(video_args),
            desc="Processing shots",
            unit="shot"
        ))
    
    # Count results
    for status, renamed_count, processed_count, failed_count, failed_list in results:
        total_renamed += renamed_count
        total_processed += processed_count
        total_failed += failed_count
        failed_files.extend(failed_list)
        
        if status == "processed":
            processed_shots += 1
        elif status == "skipped":
            skipped_shots += 1
        elif status == "failed":
            failed_shots += 1
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total shot directories: {len(video_names)}")
    print(f"  - Processed: {processed_shots}")
    print(f"  - Skipped (already exists): {skipped_shots}")
    print(f"  - Failed: {failed_shots}")
    print()
    print(f"Video operations:")
    print(f"  - Renamed to *-full.mp4: {total_renamed}")
    print(f"  - Created *-partial.mp4: {total_processed}")
    print(f"  - Failed operations: {total_failed}")
    
    if failed_files:
        print()
        print("Failed files:")
        for filename in failed_files[:20]:  # Show first 20
            print(f"  - {filename}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cut all test videos to a specified fraction of their duration"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ds2",
        help="Dataset name (e.g., 'ds2')",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for the project (default: current directory)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of video to retain (e.g., 0.3 for first 30%%, default: 0.5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-cut videos even if output file already exists",
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
    print(f"Fraction: {args.fraction} (retaining first {args.fraction*100:.1f}%%)")
    print("Operations:")
    print("  1. Rename all .mp4 files to *-full.mp4")
    print("  2. Create *-partial.mp4 files from each *-full.mp4")
    if args.force:
        print("Force mode: Will re-cut existing videos")
    print()
    
    try:
        precut_videos(
            args.dataset,
            args.base_dir,
            skip_existing=not args.force,
            num_workers=args.num_workers,
            fraction=args.fraction
        )
        print("✓ All videos processed successfully!")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

