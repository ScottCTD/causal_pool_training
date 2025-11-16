"""
Tests for metrics.py functions.

Tests cover:
- calculate_per_option_accuracy function
- All examples from docstring
- Edge cases (empty strings, single options, etc.)
"""

import sys
import os

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from causal_pool.metrics import calculate_per_option_accuracy


class TestCalculatePerOptionAccuracy:
    """Tests for calculate_per_option_accuracy function."""
    
    def test_example_1_partial_match(self):
        """Test: num_options=4, label='AB', pred='AC' -> 2/4 (docstring says 3/4 but actual is 0.5)"""
        result = calculate_per_option_accuracy(4, "AB", "AC")
        assert result == 0.5  # 2/4 - A correct, B wrong, C wrong, D correct
        
    def test_example_2_superset(self):
        """Test: num_options=4, label='AB', pred='ABC' -> 3/4"""
        result = calculate_per_option_accuracy(4, "AB", "ABC")
        assert result == 0.75  # 3/4
        
    def test_example_3_no_match(self):
        """Test: num_options=4, label='AB', pred='CD' -> 0"""
        result = calculate_per_option_accuracy(4, "AB", "CD")
        assert result == 0.0
        
    def test_example_4_perfect_match(self):
        """Test: num_options=4, label='AB', pred='AB' -> 1"""
        result = calculate_per_option_accuracy(4, "AB", "AB")
        assert result == 1.0
        
    def test_example_5_subset(self):
        """Test: num_options=4, label='AB', pred='A' -> 3/4"""
        result = calculate_per_option_accuracy(4, "AB", "A")
        assert result == 0.75  # 3/4
        
    def test_example_6_all_options(self):
        """Test: num_options=4, label='AB', pred='ABCD' -> 2/4"""
        result = calculate_per_option_accuracy(4, "AB", "ABCD")
        assert result == 0.5  # 2/4
        
    def test_duplicate_in_prediction(self):
        """Test that duplicate letters in prediction return 0"""
        result = calculate_per_option_accuracy(4, "AB", "AA")
        assert result == 0.0
        
        result = calculate_per_option_accuracy(4, "AB", "ABA")
        assert result == 0.0
        
    def test_empty_prediction(self):
        """Test empty prediction string"""
        result = calculate_per_option_accuracy(4, "AB", "")
        assert result == 0.5  # [0,0,0,0] vs [1,1,0,0] -> 2/4
        
    def test_empty_label(self):
        """Test empty label string"""
        result = calculate_per_option_accuracy(4, "", "AB")
        assert result == 0.5  # [0,0,0,0] vs [1,1,0,0] -> 2/4
        
    def test_both_empty(self):
        """Test both label and prediction empty"""
        result = calculate_per_option_accuracy(4, "", "")
        assert result == 1.0  # [0,0,0,0] vs [0,0,0,0] -> 4/4
        
    def test_single_option(self):
        """Test with single option"""
        result = calculate_per_option_accuracy(1, "A", "A")
        assert result == 1.0
        
        result = calculate_per_option_accuracy(1, "A", "")
        assert result == 0.0
        
    def test_five_options(self):
        """Test with 5 options"""
        result = calculate_per_option_accuracy(5, "ACE", "ABCD")
        # label: [1,0,1,0,1], pred: [1,1,1,1,0]
        # match: [1,0,1,0,0] -> 2/5
        assert result == 0.4
        
    def test_case_insensitive(self):
        """Test that function handles lowercase letters"""
        # letter_to_index converts to uppercase, so lowercase should work
        result = calculate_per_option_accuracy(4, "ab", "AB")
        assert result == 1.0
        
        result = calculate_per_option_accuracy(4, "aBc", "AbC")
        assert result == 1.0
        
    def test_all_options_selected(self):
        """Test when all options are selected in label"""
        result = calculate_per_option_accuracy(4, "ABCD", "AB")
        # label: [1,1,1,1], pred: [1,1,0,0]
        # match: [1,1,0,0] -> 2/4
        assert result == 0.5
        
    def test_single_correct_option(self):
        """Test with single correct option"""
        result = calculate_per_option_accuracy(4, "A", "A")
        assert result == 1.0
        
        result = calculate_per_option_accuracy(4, "A", "B")
        assert result == 0.5  # [1,0,0,0] vs [0,1,0,0] -> A wrong, B wrong, C correct, D correct -> 2/4
        
    def test_three_options_selected(self):
        """Test with three options selected"""
        result = calculate_per_option_accuracy(4, "ABC", "ABD")
        # label: [1,1,1,0], pred: [1,1,0,1]
        # match: [1,1,0,0] -> 2/4
        assert result == 0.5

