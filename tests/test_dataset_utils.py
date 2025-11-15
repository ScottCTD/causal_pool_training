"""
Tests for dataset_utils.py functions.

Tests cover:
- gather_test_dataset function
- Dataset loading and concatenation
- Size limits and shuffling
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from causal_pool.data.dataset_utils import gather_test_dataset


class TestGatherTestDataset:
    """Tests for gather_test_dataset function."""
    
    def test_gather_test_dataset_single_dataset(self, tmp_path):
        """Test loading a single dataset."""
        # Create temporary dataset structure
        dataset_name = "test_dataset"
        dataset_dir = tmp_path / "datasets" / dataset_name / "splits"
        dataset_dir.mkdir(parents=True)
        
        # Create test data
        entries = [
            {"video": f"video_{i}", "question": f"question_{i}", "ground_truth": ["A"]}
            for i in range(10)
        ]
        
        # Write to file
        import jsonlines
        with jsonlines.open(dataset_dir / "counterfactual_test.jsonl", mode='w') as writer:
            writer.write_all(entries)
        
        # Change to tmp_path directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Test loading all entries
            dataset = gather_test_dataset(dataset_name, {"counterfactual_test": 10}, random_seed=42)
            assert len(dataset) == 10
            
            # Test loading subset
            dataset = gather_test_dataset(dataset_name, {"counterfactual_test": 5}, random_seed=42)
            assert len(dataset) == 5
        finally:
            os.chdir(original_cwd)
    
    def test_gather_test_dataset_multiple_datasets(self, tmp_path):
        """Test loading and concatenating multiple datasets."""
        # Create temporary dataset structure
        dataset_name = "test_dataset"
        dataset_dir = tmp_path / "datasets" / dataset_name / "splits"
        dataset_dir.mkdir(parents=True)
        
        # Create test data for multiple datasets
        import jsonlines
        
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
        
        # Change to tmp_path directory
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Test loading all three datasets
            dataset = gather_test_dataset(
                dataset_name,
                {
                    "counterfactual_test": 10,
                    "descriptive": 8,
                    "predictive": 6
                },
                random_seed=42
            )
            assert len(dataset) == 24  # 10 + 8 + 6
            
            # Test loading subsets
            dataset = gather_test_dataset(
                dataset_name,
                {
                    "counterfactual_test": 5,
                    "descriptive": 3,
                    "predictive": 2
                },
                random_seed=42
            )
            assert len(dataset) == 10  # 5 + 3 + 2
            
            # Test loading only two datasets
            dataset = gather_test_dataset(
                dataset_name,
                {
                    "counterfactual_test": 5,
                    "descriptive": 3
                },
                random_seed=42
            )
            assert len(dataset) == 8  # 5 + 3
        finally:
            os.chdir(original_cwd)
    
    def test_gather_test_dataset_invalid_name(self, tmp_path):
        """Test that invalid dataset names raise ValueError."""
        # Create temporary dataset structure
        dataset_name = "test_dataset"
        dataset_dir = tmp_path / "datasets" / dataset_name / "splits"
        dataset_dir.mkdir(parents=True)
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Test with invalid dataset name
            with pytest.raises(ValueError, match="Invalid dataset name"):
                gather_test_dataset(dataset_name, {"invalid_name": 10}, random_seed=42)
        finally:
            os.chdir(original_cwd)
    
    def test_gather_test_dataset_shuffling(self, tmp_path):
        """Test that datasets are shuffled consistently with the same seed."""
        # Create temporary dataset structure
        dataset_name = "test_dataset"
        dataset_dir = tmp_path / "datasets" / dataset_name / "splits"
        dataset_dir.mkdir(parents=True)
        
        # Create test data
        import jsonlines
        entries = [
            {"video": f"video_{i}", "question": f"question_{i}", "ground_truth": ["A"]}
            for i in range(10)
        ]
        with jsonlines.open(dataset_dir / "counterfactual_test.jsonl", mode='w') as writer:
            writer.write_all(entries)
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Load with same seed twice - should get same order
            dataset1 = gather_test_dataset(dataset_name, {"counterfactual_test": 10}, random_seed=42)
            dataset2 = gather_test_dataset(dataset_name, {"counterfactual_test": 10}, random_seed=42)
            
            # Extract video names to compare order
            videos1 = [item["video"] for item in dataset1]
            videos2 = [item["video"] for item in dataset2]
            
            assert videos1 == videos2, "Same seed should produce same order"
            
            # Load with different seed - should get different order (likely)
            dataset3 = gather_test_dataset(dataset_name, {"counterfactual_test": 10}, random_seed=123)
            videos3 = [item["video"] for item in dataset3]
            
            # With different seed, order should likely be different
            # (small chance it could be the same, but very unlikely)
            assert videos1 != videos3 or len(set(videos1)) < len(videos1), \
                "Different seed should produce different order (or dataset has duplicates)"
        finally:
            os.chdir(original_cwd)
    
    def test_gather_test_dataset_size_limit(self, tmp_path):
        """Test that size limits are respected."""
        # Create temporary dataset structure
        dataset_name = "test_dataset"
        dataset_dir = tmp_path / "datasets" / dataset_name / "splits"
        dataset_dir.mkdir(parents=True)
        
        # Create test data with more entries than requested
        import jsonlines
        entries = [
            {"video": f"video_{i}", "question": f"question_{i}", "ground_truth": ["A"]}
            for i in range(20)
        ]
        with jsonlines.open(dataset_dir / "counterfactual_test.jsonl", mode='w') as writer:
            writer.write_all(entries)
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Request fewer entries than available
            dataset = gather_test_dataset(dataset_name, {"counterfactual_test": 5}, random_seed=42)
            assert len(dataset) == 5
            
            # Request more entries than available - should get all available
            dataset = gather_test_dataset(dataset_name, {"counterfactual_test": 100}, random_seed=42)
            # Note: select(range(100)) on a dataset of size 20 will return 20 entries
            assert len(dataset) == 20
        finally:
            os.chdir(original_cwd)
    
    def test_gather_test_dataset_empty_dataset(self, tmp_path):
        """Test handling of empty dataset files."""
        # Create temporary dataset structure
        dataset_name = "test_dataset"
        dataset_dir = tmp_path / "datasets" / dataset_name / "splits"
        dataset_dir.mkdir(parents=True)
        
        # Create empty file
        import jsonlines
        with jsonlines.open(dataset_dir / "counterfactual_test.jsonl", mode='w') as writer:
            pass  # Empty file
        
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            
            # Should handle empty dataset gracefully
            dataset = gather_test_dataset(dataset_name, {"counterfactual_test": 10}, random_seed=42)
            assert len(dataset) == 0
        finally:
            os.chdir(original_cwd)

