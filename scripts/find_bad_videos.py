#!/usr/bin/env python3
"""
Find videos with insufficient frames (< 2 frames) that will cause errors during training.

Qwen VL utils requires at least 2 frames per video. Videos with only 1 frame will cause:
ValueError: nframes should in interval [2, 1], but got 0.
"""

import argparse
import json
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


def get_video_frame_count(video_path: str) -> Tuple[int, str]:
    """
    Get video frame count using ffprobe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Tuple of (frame_count, error_message)
        frame_count: Number of frames, or -1 if error
        error_message: Empty string if successful, error message otherwise
    """
    if not os.path.exists(video_path):
        return -1, "File not found"
    
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                video_path
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        frame_count_str = result.stdout.strip()
        if frame_count_str:
            return int(frame_count_str), ""
        else:
            return -1, "No frame count returned"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        return -1, f"ffprobe error: {error_msg[:100]}"
    except ValueError:
        return -1, "Invalid frame count format"
    except FileNotFoundError:
        return -1, "ffprobe not found"


def check_single_shot(args_tuple: Tuple[str, str, int, bool]) -> List[Dict]:
    """
    Worker function to check all videos in a single shot directory.
    
    Checks all *-full.mp4 and *-partial.mp4 files in the shot directory.
    
    Args:
        args_tuple: Tuple of (shot_dir, shots_dir, min_frames, check_partial)
            - shot_dir: Name of the shot directory
            - shots_dir: Directory containing all shot directories
            - min_frames: Minimum number of frames required
            - check_partial: Whether to also check *-partial.mp4 files
    
    Returns:
        List of dictionaries with video information for bad videos (empty list if all good)
    """
    shot_dir, shots_dir, min_frames, check_partial = args_tuple
    shot_path = os.path.join(shots_dir, shot_dir)
    
    if not os.path.exists(shot_path):
        return [{
            "video": shot_dir,
            "video_file": None,
            "video_path": shot_path,
            "frame_count": -1,
            "error": "Shot directory not found",
            "status": "missing"
        }]
    
    bad_videos = []
    
    try:
        # Find all video files in the shot directory
        all_files = os.listdir(shot_path)
        
        # Check *-full.mp4 files
        full_videos = [f for f in all_files if f.endswith('-full.mp4')]
        for video_file in full_videos:
            video_path = os.path.join(shot_path, video_file)
            frame_count, error_msg = get_video_frame_count(video_path)
            
            if frame_count < min_frames:
                bad_videos.append({
                    "video": shot_dir,
                    "video_file": video_file,
                    "video_path": video_path,
                    "frame_count": frame_count,
                    "error": error_msg,
                    "status": "missing" if frame_count == -1 and "not found" in error_msg.lower() else "insufficient_frames"
                })
        
        # Optionally check *-partial.mp4 files
        if check_partial:
            partial_videos = [f for f in all_files if f.endswith('-partial.mp4')]
            for video_file in partial_videos:
                video_path = os.path.join(shot_path, video_file)
                frame_count, error_msg = get_video_frame_count(video_path)
                
                if frame_count < min_frames:
                    bad_videos.append({
                        "video": shot_dir,
                        "video_file": video_file,
                        "video_path": video_path,
                        "frame_count": frame_count,
                        "error": error_msg,
                        "status": "missing" if frame_count == -1 and "not found" in error_msg.lower() else "insufficient_frames"
                    })
        
        # If no *-full.mp4 files found, check for legacy video.mp4
        if not full_videos and not check_partial:
            legacy_video_path = os.path.join(shot_path, "video.mp4")
            if os.path.exists(legacy_video_path):
                frame_count, error_msg = get_video_frame_count(legacy_video_path)
                if frame_count < min_frames:
                    bad_videos.append({
                        "video": shot_dir,
                        "video_file": "video.mp4",
                        "video_path": legacy_video_path,
                        "frame_count": frame_count,
                        "error": error_msg,
                        "status": "missing" if frame_count == -1 and "not found" in error_msg.lower() else "insufficient_frames"
                    })
    
    except Exception as e:
        bad_videos.append({
            "video": shot_dir,
            "video_file": None,
            "video_path": shot_path,
            "frame_count": -1,
            "error": f"Error scanning directory: {str(e)}",
            "status": "missing"
        })
    
    return bad_videos


def find_bad_videos(
    dataset_name: str, 
    base_dir: str = ".", 
    min_frames: int = 2, 
    num_processes: int = 32,
    check_partial: bool = False
) -> List[Dict]:
    """
    Find all videos with insufficient frames.
    
    Checks all *-full.mp4 files in each shot directory, and optionally *-partial.mp4 files.
    
    Args:
        dataset_name: Name of the dataset (e.g., '1k_simple')
        base_dir: Base directory for the project
        min_frames: Minimum number of frames required (default: 2)
        num_processes: Number of parallel processes to use (default: 32)
        check_partial: Whether to also check *-partial.mp4 files (default: False)
    
    Returns:
        List of dictionaries with video information and issues
    """
    dataset_base_path = os.path.join(base_dir, "datasets", dataset_name)
    shots_dir = os.path.join(dataset_base_path, "shots")
    
    if not os.path.exists(shots_dir):
        raise FileNotFoundError(f"Shots directory not found: {shots_dir}")
    
    # Scan all shot directories
    shot_dirs = sorted([d for d in os.listdir(shots_dir) if os.path.isdir(os.path.join(shots_dir, d))])
    
    check_type = "*-full.mp4 and *-partial.mp4" if check_partial else "*-full.mp4"
    print(f"Scanning {len(shot_dirs)} shot directories for {check_type} files with {num_processes} processes...")
    
    # Prepare arguments for worker function
    work_items = [(shot_dir, shots_dir, min_frames, check_partial) for shot_dir in shot_dirs]
    
    # Process shot directories in parallel
    bad_videos = []
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(check_single_shot, work_items),
            total=len(work_items),
            desc="Checking shots"
        ))
    
    # Flatten results (each shot can have multiple bad videos)
    for shot_bad_videos in results:
        bad_videos.extend(shot_bad_videos)
    
    return bad_videos


def main():
    parser = argparse.ArgumentParser(
        description="Find videos with insufficient frames (< 2 frames) that will cause training errors"
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
        "--min-frames",
        type=int,
        default=2,
        help="Minimum number of frames required (default: 2, as required by Qwen VL utils)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file to save bad videos list (default: datasets/<dataset>/bad_videos.json)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, not individual video details",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=32,
        help="Number of parallel processes to use (default: 32)",
    )
    parser.add_argument(
        "--check-partial",
        action="store_true",
        help="Also check *-partial.mp4 files (default: only check *-full.mp4)",
    )
    
    args = parser.parse_args()
    
    # Set default output path under dataset directory if not specified
    if args.output is None:
        dataset_base_path = os.path.join(args.base_dir, "datasets", args.dataset)
        args.output = os.path.join(dataset_base_path, "bad_videos.json")
    
    print(f"Finding bad videos in dataset: {args.dataset}")
    print(f"Minimum frames required: {args.min_frames}")
    print(f"Using {args.processes} parallel processes")
    if args.check_partial:
        print("Checking both *-full.mp4 and *-partial.mp4 files")
    else:
        print("Checking *-full.mp4 files only (use --check-partial to also check *-partial.mp4)")
    print()
    
    bad_videos = find_bad_videos(
        args.dataset, 
        args.base_dir, 
        args.min_frames, 
        args.processes,
        args.check_partial
    )
    
    # Group by status
    missing = [v for v in bad_videos if v["status"] == "missing"]
    insufficient = [v for v in bad_videos if v["status"] == "insufficient_frames"]
    
    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total bad videos found: {len(bad_videos)}")
    print(f"  - Missing files: {len(missing)}")
    print(f"  - Insufficient frames (< {args.min_frames}): {len(insufficient)}")
    print()
    
    if not args.summary_only:
        if missing:
            print("MISSING VIDEOS:")
            print("-" * 70)
            for video in missing:
                video_file_str = f"/{video['video_file']}" if video.get('video_file') else ""
                print(f"  {video['video']}{video_file_str}: {video['error']}")
            print()
        
        if insufficient:
            print(f"VIDEOS WITH INSUFFICIENT FRAMES (< {args.min_frames}):")
            print("-" * 70)
            for video in insufficient:
                video_file_str = f"/{video['video_file']}" if video.get('video_file') else ""
                print(f"  {video['video']}{video_file_str}: {video['frame_count']} frame(s) - {video['error']}")
            print()
        
        if bad_videos:
            print("ALL BAD VIDEOS (sorted by shot directory and filename):")
            print("-" * 70)
            for video in sorted(bad_videos, key=lambda x: (x['video'], x.get('video_file', ''))):
                status_str = f"[{video['status']}]"
                video_file_str = f"/{video['video_file']}" if video.get('video_file') else ""
                if video['frame_count'] >= 0:
                    print(f"  {video['video']}{video_file_str:<30} {status_str:<20} {video['frame_count']} frame(s)")
                else:
                    print(f"  {video['video']}{video_file_str:<30} {status_str:<20} {video['error']}")
    
    # Save to file if requested
    if args.output:
        output_data = {
            "dataset": args.dataset,
            "min_frames": args.min_frames,
            "total_bad_videos": len(bad_videos),
            "missing_count": len(missing),
            "insufficient_frames_count": len(insufficient),
            "bad_videos": bad_videos,
        }
        
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with error code if bad videos found
    if bad_videos:
        print(f"\n⚠️  WARNING: {len(bad_videos)} bad video(s) found!")
        return 1
    else:
        print("\n✓ All videos are valid!")
        return 0


if __name__ == "__main__":
    exit(main())


