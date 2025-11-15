#!/usr/bin/env python3
"""
Manual test script for gather_test_dataset function.
This can be run without pytest to verify the function works.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from causal_pool.data.dataset_utils import gather_test_dataset
import jsonlines


def test_gather_test_dataset():
    """Manual test for gather_test_dataset."""
    print("Testing gather_test_dataset function...")
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_name = "test_dataset"
        dataset_dir = tmp_path / "datasets" / dataset_name / "splits"
        dataset_dir.mkdir(parents=True)
        
        # Create test data
        counterfactual_entries = [
            {"video": f"cf_video_{i}", "question": f"cf_question_{i}", "ground_truth": ["A"]}
            for i in range(10)
        ]
        with jsonlines.open(dataset_dir / "counterfactual_test.jsonl", mode='w') as writer:
            writer.write_all(counterfactual_entries)
        
        descriptive_entries = [
            {"video": f"desc_video_{i}", "question": f"desc_question_{i}", "ground_truth": ["B"]}
            for i in range(8)
        ]
        with jsonlines.open(dataset_dir / "descriptive.jsonl", mode='w') as writer:
            writer.write_all(descriptive_entries)
        
        predictive_entries = [
            {"video": f"pred_video_{i}", "question": f"pred_question_{i}", "ground_truth": ["C"]}
            for i in range(6)
        ]
        with jsonlines.open(dataset_dir / "predictive.jsonl", mode='w') as writer:
            writer.write_all(predictive_entries)
        
        # Change to tmpdir
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Test 1: Single dataset
            print("\nTest 1: Loading single dataset (counterfactual_test)")
            dataset = gather_test_dataset(dataset_name, {"counterfactual_test": 10}, random_seed=42)
            assert len(dataset) == 10, f"Expected 10 entries, got {len(dataset)}"
            print(f"✓ Loaded {len(dataset)} entries")
            
            # Test 2: Multiple datasets
            print("\nTest 2: Loading multiple datasets")
            dataset = gather_test_dataset(
                dataset_name,
                {
                    "counterfactual_test": 10,
                    "descriptive": 8,
                    "predictive": 6
                },
                random_seed=42
            )
            assert len(dataset) == 24, f"Expected 24 entries, got {len(dataset)}"
            print(f"✓ Loaded {len(dataset)} entries (10 + 8 + 6)")
            
            # Test 3: Size limits
            print("\nTest 3: Testing size limits")
            dataset = gather_test_dataset(dataset_name, {"counterfactual_test": 5}, random_seed=42)
            assert len(dataset) == 5, f"Expected 5 entries, got {len(dataset)}"
            print(f"✓ Loaded {len(dataset)} entries (limited to 5)")
            
            # Test 4: Shuffling consistency
            print("\nTest 4: Testing shuffling consistency")
            dataset1 = gather_test_dataset(dataset_name, {"counterfactual_test": 10}, random_seed=42)
            dataset2 = gather_test_dataset(dataset_name, {"counterfactual_test": 10}, random_seed=42)
            videos1 = [item["video"] for item in dataset1]
            videos2 = [item["video"] for item in dataset2]
            assert videos1 == videos2, "Same seed should produce same order"
            print("✓ Same seed produces same order")
            
            # Test 5: Invalid dataset name
            print("\nTest 5: Testing invalid dataset name")
            try:
                gather_test_dataset(dataset_name, {"invalid_name": 10}, random_seed=42)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Invalid dataset name" in str(e)
                print("✓ Invalid dataset name correctly raises ValueError")
            
            print("\n" + "="*50)
            print("All tests passed! ✓")
            print("="*50)
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_gather_test_dataset()

