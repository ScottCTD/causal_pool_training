"""
Utility functions for evaluation script.

This module contains:
- Video processing utilities (duration, cutting)
- Model configuration utilities (hyperparameters, normalization)
- Evaluation utilities (metrics calculation, prediction validation)
- Prompt building utilities
"""

import base64
import os
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any
from causal_pool.prompt_utils import build_question_prompt


class InvalidPredictionError(ValueError):
    """Raised when prediction format is invalid (not pure A-Z or has duplicates)."""
    pass


# Model-specific hyperparameter defaults
MODEL_HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen3-VL-4B-Instruct": {
        "temperature": 0.8,
        "top_k": 20,
        "top_p": 0.8,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
    },
    "Qwen/Qwen3-VL-4B-Thinking": {
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "temperature": 0.8,
        "top_k": 20,
        "top_p": 0.8,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.5,
    },
    "OpenGVLab/InternVL3_5-4B": {
        "temperature": 0.0,
    }
}

# Default hyperparameters (used if model not found in MODEL_HYPERPARAMETERS)
DEFAULT_HYPERPARAMETERS = {
    "temperature": 0.8,
    "top_k": 20,
    "top_p": 0.8,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
}


def normalize_model_name(model_name: str) -> str:
    """Normalize model name for file naming."""
    return model_name.split("/")[-1].replace("-", "_").lower()


def get_model_hyperparameters(model_name: str) -> Dict[str, Any]:
    """
    Get hyperparameters for a specific model, falling back to defaults if not found.
    
    Args:
        model_name: Model name (e.g., "Qwen/Qwen3-VL-4B-Instruct")
    
    Returns:
        Dictionary of hyperparameters
    """
    return MODEL_HYPERPARAMETERS.get(model_name, DEFAULT_HYPERPARAMETERS).copy()


def get_metrics(entry: Dict[str, Any], pred: str) -> Tuple[int, int]:
    """
    Returns (exactly correct or not, how many options were correct).
    
    Args:
        entry: Dataset entry with 'ground_truth' field
        pred: Prediction string (e.g., "AC")
    
    Returns:
        Tuple of (exactly_correct, num_correct_options)
    
    Raises:
        InvalidPredictionError: If prediction format is invalid (not pure A-Z or has duplicates)
    """
    if (idx := pred.rfind("</think>")) != -1:
        pred = pred[idx + len("</think>"):]
    
    pred = pred.strip()
    
    if not all(c.isalpha() and c.isupper() for c in pred):
        raise InvalidPredictionError(f"Prediction contains non-A-Z characters: {pred!r}")
    
    selected_options = set(ord(c) - ord("A") for c in pred)

    if len(selected_options) != len(pred):  # duplicate options
        raise InvalidPredictionError(f"Prediction contains duplicate options: {pred!r}")
    
    ground_truth = set(entry["ground_truth"])
    
    exactly_correct = int(selected_options == ground_truth)
    num_correct = len(selected_options & ground_truth)
    
    return exactly_correct, num_correct


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Duration in seconds
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to get video duration for {video_path}: {e}")


def cut_video_first_half(video_path: str, output_path: str) -> None:
    """
    Cut video to first half using ffmpeg.
    
    First tries codec copy (fast), falls back to re-encoding if that fails.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save cut video
    """
    duration = get_video_duration(video_path)
    half_duration = duration / 2.0
    
    # First try codec copy (fast, no re-encoding)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-t", str(half_duration),
                "-c", "copy",  # Copy codec to avoid re-encoding (faster)
                "-y",  # Overwrite output file
                output_path
            ],
            capture_output=True,
            check=True,
        )
        return  # Success with codec copy
    except subprocess.CalledProcessError:
        # Codec copy failed (e.g., cutting at non-keyframe), fall through to re-encoding
        pass
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to use predictive question type.")
    
    # Fall back to re-encoding if codec copy fails (e.g., cutting at non-keyframe)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-t", str(half_duration),
                "-c:v", "libx264",  # Re-encode video
                "-c:a", "aac",  # Re-encode audio
                "-preset", "fast",  # Faster encoding
                "-y",  # Overwrite output file
                output_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"Failed to cut video {video_path}: {error_msg}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to use predictive question type.")


def build_prompt(entry: Dict[str, Any], dataset_name: str, base_dir: str = ".") -> List[Dict[str, Any]]:
    """
    Build prompt for a dataset entry.
    
    For predictive questions, only the first half of the video is used.
    
    Args:
        entry: Dataset entry
        dataset_name: Name of the dataset
        base_dir: Base directory for the project
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    video_path = os.path.join(
        base_dir, "datasets", dataset_name, "shots", entry["video"], f"video.mp4"
    )
    
    question_prompt = build_question_prompt(entry)

    # Check if this is a predictive question type
    question_type = entry.get("metadata", {}).get("question_type", "")
    is_predictive = question_type == "predictive"
    
    # For predictive questions, cut video to first half
    video_to_encode = video_path
    temp_video_path = None
    
    if is_predictive:
        # Create temporary file for cut video
        temp_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)
        try:
            cut_video_first_half(video_path, temp_video_path)
            video_to_encode = temp_video_path
        except Exception as e:
            # Clean up temp file if cutting fails
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            raise RuntimeError(f"Failed to process predictive video: {e}")
    
    # Read and encode video
    try:
        with open(video_to_encode, "rb") as video_file:
            video_b64 = base64.b64encode(video_file.read()).decode("utf-8")
    finally:
        # Clean up temporary cut video if it was created
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
    
    return [{
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
            },
            {"type": "text", "text": question_prompt},
        ],
    }]

