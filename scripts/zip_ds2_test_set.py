#!/usr/bin/env python3
"""
Zip ds2 test set including all test QA files and all referenced video shots
(including counterfactual video shots).
"""

import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Set

import jsonlines


def collect_test_shots(dataset_dir: Path) -> Set[str]:
    """
    Collect all shot IDs referenced in test QA files.
    Returns a set of shot IDs (e.g., {"shot_1", "shot_1469", ...}).
    """
    shots = set()
    splits_dir = dataset_dir / "splits"
    
    # All test QA files
    test_files = [
        "test-descriptive.jsonl",
        "test-predictive.jsonl",
        "test-counterfactual_position.jsonl",
        "test-counterfactual_velocity.jsonl",
    ]
    
    for test_file in test_files:
        test_path = splits_dir / test_file
        if not test_path.exists():
            print(f"Warning: {test_file} not found, skipping...")
            continue
        
        print(f"Processing {test_file}...")
        with jsonlines.open(test_path) as reader:
            for entry in reader:
                # Add the main video shot
                video = entry.get("video")
                if video:
                    shots.add(video)
                
                # Add counterfactual video shot if present
                metadata = entry.get("metadata", {})
                counterfactual_video = metadata.get("counterfactual_video")
                if counterfactual_video:
                    shots.add(counterfactual_video)
    
    return shots


def zip_test_set(dataset_dir: Path, output_zip: Path):
    """
    Create a zip file containing:
    1. All test QA files
    2. All shot directories referenced in test QA files
    """
    print(f"Collecting test shots from {dataset_dir}...")
    shots = collect_test_shots(dataset_dir)
    print(f"Found {len(shots)} unique shot IDs")
    
    splits_dir = dataset_dir / "splits"
    shots_dir = dataset_dir / "shots"
    
    # Test QA files to include
    test_files = [
        "test-descriptive.jsonl",
        "test-predictive.jsonl",
        "test-counterfactual_position.jsonl",
        "test-counterfactual_velocity.jsonl",
    ]
    
    print(f"\nCreating zip file: {output_zip}")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all test QA files
        print("\nAdding test QA files...")
        for test_file in test_files:
            test_path = splits_dir / test_file
            if test_path.exists():
                zipf.write(test_path, f"splits/{test_file}")
                print(f"  Added splits/{test_file}")
            else:
                print(f"  Warning: {test_file} not found, skipping...")
        
        # Add all referenced shot directories
        print(f"\nAdding {len(shots)} shot directories...")
        missing_shots = []
        added_count = 0
        
        for shot_id in sorted(shots):
            shot_dir = shots_dir / shot_id
            if not shot_dir.exists():
                missing_shots.append(shot_id)
                continue
            
            # Add all files in the shot directory
            for file_path in shot_dir.rglob('*'):
                if file_path.is_file():
                    arcname = f"shots/{shot_id}/{file_path.relative_to(shot_dir)}"
                    zipf.write(file_path, arcname)
            
            added_count += 1
            if added_count % 100 == 0:
                print(f"  Added {added_count}/{len(shots)} shots...")
        
        if missing_shots:
            print(f"\nWarning: {len(missing_shots)} shot directories not found:")
            for shot_id in missing_shots[:10]:  # Show first 10
                print(f"  {shot_id}")
            if len(missing_shots) > 10:
                print(f"  ... and {len(missing_shots) - 10} more")
        
        print(f"\nAdded {added_count} shot directories")
    
    print(f"\nZip file created: {output_zip}")
    print(f"File size: {output_zip.stat().st_size / (1024**3):.2f} GB")


def main():
    """Main function."""
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "datasets" / "ds2"
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        sys.exit(1)
    
    output_zip = base_dir / "ds2_test_set.zip"
    
    # Remove existing zip if it exists
    if output_zip.exists():
        print(f"Removing existing zip file: {output_zip}")
        output_zip.unlink()
    
    zip_test_set(dataset_dir, output_zip)


if __name__ == "__main__":
    main()

