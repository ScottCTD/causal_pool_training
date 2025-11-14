"""
Comprehensive tests for eval.py evaluation script.

Tests cover:
- Model name normalization
- Metrics calculation
- Video processing (duration, cutting)
- Prompt building with question types
- Per-question-type metrics tracking
- Full evaluation flow
"""

import asyncio
import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import functions from eval.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval.eval import (
    normalize_model_name,
    get_metrics,
    get_video_duration,
    cut_video_first_half,
    build_prompt,
    AsyncEvaluator,
    evaluate_dataset,
)


class TestNormalizeModelName:
    """Tests for normalize_model_name function."""
    
    def test_simple_model_name(self):
        assert normalize_model_name("Qwen/Qwen3-VL-4B-Instruct") == "qwen3_vl_4b_instruct"
    
    def test_model_name_with_slashes(self):
        assert normalize_model_name("org/model-name") == "model_name"
    
    def test_model_name_with_dashes(self):
        assert normalize_model_name("test-model") == "test_model"
    
    def test_model_name_lowercase(self):
        assert normalize_model_name("MODEL-NAME") == "model_name"


class TestGetMetrics:
    """Tests for get_metrics function."""
    
    def test_exactly_correct_single_option(self):
        entry = {"ground_truth": [0]}
        pred = "A"
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 1
        assert num_correct == 1
    
    def test_exactly_correct_multiple_options(self):
        entry = {"ground_truth": [0, 2]}
        pred = "AC"
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 1
        assert num_correct == 2
    
    def test_partially_correct(self):
        entry = {"ground_truth": [0, 2, 3]}
        pred = "AC"  # Only 2 out of 3 correct
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 0
        assert num_correct == 2
    
    def test_wrong_prediction(self):
        entry = {"ground_truth": [0, 1]}
        pred = "CD"  # None correct
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 0
        assert num_correct == 0
    
    def test_extra_options(self):
        entry = {"ground_truth": [0]}
        pred = "ABC"  # Extra options
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 0
        assert num_correct == 1
    
    def test_duplicate_options(self):
        entry = {"ground_truth": [0]}
        pred = "AA"  # Duplicate
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 0
        assert num_correct == 0
    
    def test_lowercase_prediction(self):
        entry = {"ground_truth": [0]}
        pred = "a"  # Lowercase
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 0
        assert num_correct == 0
    
    def test_non_alpha_prediction(self):
        entry = {"ground_truth": [0]}
        pred = "A1"  # Contains number
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 0
        assert num_correct == 0
    
    def test_empty_prediction(self):
        entry = {"ground_truth": [0]}
        pred = ""
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 0
        assert num_correct == 0
    
    def test_whitespace_handling(self):
        entry = {"ground_truth": [0, 1]}
        pred = "  AB  "  # With whitespace
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 1
        assert num_correct == 2
    
    def test_all_options(self):
        entry = {"ground_truth": [0, 1, 2, 3, 4, 5]}
        pred = "ABCDEF"
        exactly_correct, num_correct = get_metrics(entry, pred)
        assert exactly_correct == 1
        assert num_correct == 6


class TestGetVideoDuration:
    """Tests for get_video_duration function."""
    
    @patch('subprocess.run')
    def test_successful_duration_extraction(self, mock_run):
        mock_run.return_value = Mock(stdout="10.5\n", returncode=0)
        duration = get_video_duration("/path/to/video.mp4")
        assert duration == 10.5
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_duration_with_decimal(self, mock_run):
        mock_run.return_value = Mock(stdout="123.456\n", returncode=0)
        duration = get_video_duration("/path/to/video.mp4")
        assert duration == 123.456
    
    @patch('subprocess.run')
    def test_ffprobe_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(RuntimeError, match="Failed to get video duration"):
            get_video_duration("/path/to/video.mp4")
    
    @patch('subprocess.run')
    def test_ffprobe_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")
        with pytest.raises(RuntimeError, match="Failed to get video duration"):
            get_video_duration("/path/to/video.mp4")
    
    @patch('subprocess.run')
    def test_invalid_output(self, mock_run):
        mock_run.return_value = Mock(stdout="invalid\n", returncode=0)
        with pytest.raises(RuntimeError, match="Failed to get video duration"):
            get_video_duration("/path/to/video.mp4")


class TestCutVideoFirstHalf:
    """Tests for cut_video_first_half function."""
    
    @patch('eval.eval.get_video_duration')
    @patch('subprocess.run')
    def test_successful_codec_copy(self, mock_run, mock_duration):
        mock_duration.return_value = 10.0
        mock_run.return_value = Mock(returncode=0)
        
        cut_video_first_half("/path/to/input.mp4", "/path/to/output.mp4")
        
        # Should call ffmpeg with codec copy
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "-c" in args
        assert "copy" in args
        assert "-t" in args
        assert "5.0" in args  # Half of 10.0
    
    @patch('eval.eval.get_video_duration')
    @patch('subprocess.run')
    def test_fallback_to_reencoding(self, mock_run, mock_duration):
        mock_duration.return_value = 8.0
        # First call fails (codec copy), second succeeds (re-encoding)
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "ffmpeg"),  # Codec copy fails
            Mock(returncode=0)  # Re-encoding succeeds
        ]
        
        cut_video_first_half("/path/to/input.mp4", "/path/to/output.mp4")
        
        # Should have tried twice
        assert mock_run.call_count == 2
        # Second call should use re-encoding
        second_call_args = mock_run.call_args_list[1][0][0]
        assert "-c:v" in second_call_args
        assert "libx264" in second_call_args
    
    @patch('eval.eval.get_video_duration')
    @patch('subprocess.run')
    def test_ffmpeg_not_found(self, mock_run, mock_duration):
        mock_duration.return_value = 10.0
        mock_run.side_effect = FileNotFoundError()
        
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            cut_video_first_half("/path/to/input.mp4", "/path/to/output.mp4")
    
    @patch('eval.eval.get_video_duration')
    @patch('subprocess.run')
    def test_both_methods_fail(self, mock_run, mock_duration):
        mock_duration.return_value = 10.0
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "ffmpeg"),  # Codec copy fails
            subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Error")  # Re-encoding fails
        ]
        
        with pytest.raises(RuntimeError, match="Failed to cut video"):
            cut_video_first_half("/path/to/input.mp4", "/path/to/output.mp4")


class TestBuildPrompt:
    """Tests for build_prompt function."""
    
    def test_descriptive_question_full_video(self, tmp_path):
        """Test that descriptive questions use full video."""
        # Create a dummy video file
        video_dir = tmp_path / "datasets" / "test_dataset" / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        entry = {
            "video": "shot_01",
            "question": "What happened?",
            "options": ["Option A", "Option B"],
            "metadata": {"question_type": "descriptive"}
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"fake video content"
            result = build_prompt(entry, "test_dataset", base_dir=str(tmp_path))
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][1]["type"] == "text"
        assert "What happened?" in result[0]["content"][1]["text"]
        assert "Option A" in result[0]["content"][1]["text"]
        assert "Option B" in result[0]["content"][1]["text"]
    
    @patch('eval.eval.cut_video_first_half')
    def test_predictive_question_cuts_video(self, mock_cut, tmp_path):
        """Test that predictive questions cut video to first half."""
        # Create a dummy video file
        video_dir = tmp_path / "datasets" / "test_dataset" / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        entry = {
            "video": "shot_01",
            "question": "What will happen?",
            "options": ["Option A", "Option B"],
            "metadata": {"question_type": "predictive"}
        }
        
        # Mock tempfile.mkstemp to return a temp path
        with patch('tempfile.mkstemp') as mock_mkstemp, \
             patch('builtins.open', create=True) as mock_open, \
             patch('os.close'), \
             patch('os.path.exists', return_value=True), \
             patch('os.unlink'):
            mock_mkstemp.return_value = (123, "/tmp/temp_video.mp4")
            mock_open.return_value.__enter__.return_value.read.return_value = b"cut video content"
            
            result = build_prompt(entry, "test_dataset", base_dir=str(tmp_path))
            
            # Should have called cut_video_first_half
            mock_cut.assert_called_once()
            assert result[0]["content"][0]["type"] == "video_url"
    
    def test_unknown_question_type_full_video(self, tmp_path):
        """Test that unknown question types use full video."""
        video_dir = tmp_path / "datasets" / "test_dataset" / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        entry = {
            "video": "shot_01",
            "question": "Question?",
            "options": ["A", "B"],
            "metadata": {}  # No question_type
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"fake video content"
            result = build_prompt(entry, "test_dataset", base_dir=str(tmp_path))
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
    
    def test_prompt_format(self, tmp_path):
        """Test that prompt is formatted correctly."""
        video_dir = tmp_path / "datasets" / "test_dataset" / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        entry = {
            "video": "shot_01",
            "question": "Test question?",
            "options": ["First option", "Second option", "Third option"],
            "metadata": {"question_type": "descriptive"}
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"fake video content"
            result = build_prompt(entry, "test_dataset", base_dir=str(tmp_path))
        
        text_content = result[0]["content"][1]["text"]
        assert "Test question?" in text_content
        assert "A. First option" in text_content
        assert "B. Second option" in text_content
        assert "C. Third option" in text_content
        assert "Please select the correct option(s)" in text_content


class TestAsyncEvaluator:
    """Tests for AsyncEvaluator class."""
    
    def test_initialization(self):
        evaluator = AsyncEvaluator(
            base_url="http://test.com",
            model="test-model",
            api_key="test-key",
            max_tokens=100,
            temperature=0.7
        )
        assert evaluator.model == "test-model"
        assert evaluator.max_tokens == 100
        assert evaluator.temperature == 0.7
        assert evaluator.max_retries == 10
    
    @pytest.mark.asyncio
    async def test_evaluate_entry_success(self, tmp_path):
        """Test successful evaluation of an entry."""
        # Create dummy video
        video_dir = tmp_path / "datasets" / "test_dataset" / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        entry = {
            "video": "shot_01",
            "question": "Test?",
            "options": ["A", "B"],
            "metadata": {"question_type": "descriptive"}
        }
        
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="A"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        evaluator = AsyncEvaluator(
            base_url="http://test.com",
            model="test-model"
        )
        evaluator.client = mock_client
        
        with patch('eval.eval.build_prompt', return_value=[{"role": "user", "content": []}]):
            result = await evaluator.evaluate_entry(entry, "test_dataset", base_dir=str(tmp_path))
        
        assert result == "A"
        mock_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_evaluate_entry_with_samples(self, tmp_path):
        """Test evaluation with multiple samples."""
        video_dir = tmp_path / "datasets" / "test_dataset" / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        entry = {
            "video": "shot_01",
            "question": "Test?",
            "options": ["A", "B"],
            "metadata": {"question_type": "descriptive"}
        }
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="A"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        evaluator = AsyncEvaluator(
            base_url="http://test.com",
            model="test-model"
        )
        evaluator.client = mock_client
        
        with patch('eval.eval.build_prompt', return_value=[{"role": "user", "content": []}]):
            results = await evaluator.evaluate_entry_with_samples(
                entry, "test_dataset", num_samples=3, base_dir=str(tmp_path)
            )
        
        assert len(results) == 3
        assert all(r == "A" for r in results)


class TestEvaluateDataset:
    """Tests for evaluate_dataset function."""
    
    @pytest.mark.asyncio
    async def test_evaluate_dataset_with_question_types(self, tmp_path):
        """Test that question types are tracked correctly."""
        # Create a test dataset JSONL file
        dataset_dir = tmp_path / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        dataset_file = dataset_dir / "test_dataset.jsonl"
        
        entries = [
            {
                "video": "shot_01",
                "question": "Descriptive question?",
                "options": ["A", "B"],
                "ground_truth": [0],
                "metadata": {"question_type": "descriptive"}
            },
            {
                "video": "shot_01",
                "question": "Predictive question?",
                "options": ["A", "B"],
                "ground_truth": [0],
                "metadata": {"question_type": "predictive"}
            },
            {
                "video": "shot_01",
                "question": "Counterfactual question?",
                "options": ["A", "B"],
                "ground_truth": [1],
                "metadata": {"question_type": "counterfactual_velocity"}
            },
        ]
        
        with open(dataset_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        # Create dummy videos
        video_dir = dataset_dir / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        # Mock evaluator
        mock_evaluator = AsyncMock()
        mock_evaluator.model = "test-model"
        mock_evaluator.evaluate_entry_with_samples = AsyncMock(
            side_effect=lambda entry, *args, **kwargs: ["A" if entry["ground_truth"][0] == 0 else "B"]
        )
        
        results = await evaluate_dataset(
            dataset_path=str(dataset_file),
            evaluator=mock_evaluator,
            num_samples=1,
            max_concurrent=1,
            base_dir=str(tmp_path)
        )
        
        # Check that results include question types
        assert "per_question_type" in results["metrics"]
        assert "descriptive" in results["metrics"]["per_question_type"]
        assert "predictive" in results["metrics"]["per_question_type"]
        assert "counterfactual_velocity" in results["metrics"]["per_question_type"]
        
        # Check detailed results include question_type
        assert len(results["detailed_results"]) == 3
        for result in results["detailed_results"]:
            assert "question_type" in result
    
    @pytest.mark.asyncio
    async def test_per_question_type_metrics(self, tmp_path):
        """Test that per-question-type metrics are calculated correctly."""
        dataset_dir = tmp_path / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        dataset_file = dataset_dir / "test_dataset.jsonl"
        
        # Create entries with different question types and ground truths
        entries = [
            {
                "video": "shot_01",
                "question": "Q1",
                "options": ["A", "B"],
                "ground_truth": [0],
                "metadata": {"question_type": "descriptive"}
            },
            {
                "video": "shot_01",
                "question": "Q2",
                "options": ["A", "B"],
                "ground_truth": [0],
                "metadata": {"question_type": "descriptive"}
            },
            {
                "video": "shot_01",
                "question": "Q3",
                "options": ["A", "B"],
                "ground_truth": [1],
                "metadata": {"question_type": "predictive"}
            },
        ]
        
        with open(dataset_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        video_dir = dataset_dir / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        # Mock evaluator to return correct predictions for first two, wrong for third
        def mock_evaluate(entry, *args, **kwargs):
            # Return "A" for descriptive (correct), "A" for predictive (wrong, should be "B")
            if entry["metadata"]["question_type"] == "descriptive":
                return ["A"]
            else:
                return ["A"]  # Wrong prediction
        
        mock_evaluator = AsyncMock()
        mock_evaluator.model = "test-model"
        mock_evaluator.evaluate_entry_with_samples = AsyncMock(side_effect=mock_evaluate)
        
        results = await evaluate_dataset(
            dataset_path=str(dataset_file),
            evaluator=mock_evaluator,
            num_samples=1,
            max_concurrent=1,
            base_dir=str(tmp_path)
        )
        
        # Check per-question-type metrics
        desc_metrics = results["metrics"]["per_question_type"]["descriptive"]
        assert desc_metrics["total_questions"] == 2
        assert desc_metrics["questions_with_exact_match"] == 2  # Both correct
        assert desc_metrics["per_question_accuracy"] == 1.0
        
        pred_metrics = results["metrics"]["per_question_type"]["predictive"]
        assert pred_metrics["total_questions"] == 1
        assert pred_metrics["questions_with_exact_match"] == 0  # Wrong prediction
        assert pred_metrics["per_question_accuracy"] == 0.0
    
    @pytest.mark.asyncio
    async def test_multiple_samples_per_question_type(self, tmp_path):
        """Test @1 and @k metrics per question type."""
        dataset_dir = tmp_path / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        dataset_file = dataset_dir / "test_dataset.jsonl"
        
        entry = {
            "video": "shot_01",
            "question": "Q1",
            "options": ["A", "B"],
            "ground_truth": [0],
            "metadata": {"question_type": "descriptive"}
        }
        
        with open(dataset_file, 'w') as f:
            f.write(json.dumps(entry) + '\n')
        
        video_dir = dataset_dir / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        # Mock evaluator: first sample wrong, second sample correct
        mock_evaluator = AsyncMock()
        mock_evaluator.model = "test-model"
        mock_evaluator.evaluate_entry_with_samples = AsyncMock(return_value=["B", "A"])
        
        results = await evaluate_dataset(
            dataset_path=str(dataset_file),
            evaluator=mock_evaluator,
            num_samples=2,
            max_concurrent=1,
            base_dir=str(tmp_path)
        )
        
        desc_metrics = results["metrics"]["per_question_type"]["descriptive"]
        assert "accuracy@1" in desc_metrics
        assert "accuracy@2" in desc_metrics
        assert desc_metrics["accuracy@1"] == 0.0  # First sample wrong
        assert desc_metrics["accuracy@2"] == 1.0  # Any sample correct
    
    @pytest.mark.asyncio
    async def test_error_handling_with_question_types(self, tmp_path):
        """Test that errors are tracked per question type."""
        dataset_dir = tmp_path / "datasets" / "test_dataset"
        dataset_dir.mkdir(parents=True)
        dataset_file = dataset_dir / "test_dataset.jsonl"
        
        entries = [
            {
                "video": "shot_01",
                "question": "Q1",
                "options": ["A", "B"],
                "ground_truth": [0],
                "metadata": {"question_type": "descriptive"}
            },
            {
                "video": "shot_01",
                "question": "Q2",
                "options": ["A", "B"],
                "ground_truth": [0],
                "metadata": {"question_type": "predictive"}
            },
        ]
        
        with open(dataset_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        video_dir = dataset_dir / "shots" / "shot_01"
        video_dir.mkdir(parents=True)
        video_file = video_dir / "video_shot_01.mp4"
        video_file.write_bytes(b"fake video content")
        
        # Mock evaluator: first succeeds, second fails
        def mock_evaluate(entry, *args, **kwargs):
            if entry["metadata"]["question_type"] == "descriptive":
                return ["A"]
            else:
                raise Exception("Test error")
        
        mock_evaluator = AsyncMock()
        mock_evaluator.model = "test-model"
        mock_evaluator.evaluate_entry_with_samples = AsyncMock(side_effect=mock_evaluate)
        
        results = await evaluate_dataset(
            dataset_path=str(dataset_file),
            evaluator=mock_evaluator,
            num_samples=1,
            max_concurrent=1,
            base_dir=str(tmp_path)
        )
        
        # Check that error is tracked
        assert len(results["detailed_results"]) == 2
        # Successful result doesn't have "error" key
        assert "error" not in results["detailed_results"][0] or results["detailed_results"][0].get("error") is None
        assert results["detailed_results"][1]["error"] == "Test error"
        
        # Check that question types are still tracked even for errors
        assert results["detailed_results"][0]["question_type"] == "descriptive"
        assert results["detailed_results"][1]["question_type"] == "predictive"
        
        # Check that error entries are counted in question type stats
        assert "descriptive" in results["metrics"]["per_question_type"]
        assert "predictive" in results["metrics"]["per_question_type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

