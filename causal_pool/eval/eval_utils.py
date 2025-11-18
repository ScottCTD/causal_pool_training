"""
Utility functions for evaluation script.

This module contains:
- Model configuration utilities (hyperparameters, normalization)
- Evaluation utilities (metrics calculation, prediction validation)
- Prompt building utilities
"""

import base64
import os
from typing import Dict, List, Tuple, Any
from causal_pool.prompt_utils import build_question_prompt
from causal_pool.utils import normalize_model_name


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


def get_model_hyperparameters(model_name: str) -> Dict[str, Any]:
    """
    Get hyperparameters for a specific model, falling back to defaults if not found.
    
    Args:
        model_name: Model name (e.g., "Qwen/Qwen3-VL-4B-Instruct")
    
    Returns:
        Dictionary of hyperparameters
    """
    return MODEL_HYPERPARAMETERS.get(model_name, DEFAULT_HYPERPARAMETERS).copy()


def build_prompt(entry: Dict[str, Any], dataset_name: str, base_dir: str = ".", additional_suffix: str = None) -> List[Dict[str, Any]]:
    """
    Build prompt for a dataset entry.
    
    For predictive questions, uses pre-cut video_partial.mp4 instead of video.mp4.
    
    Args:
        entry: Dataset entry
        dataset_name: Name of the dataset
        base_dir: Base directory for the project
        additional_suffix: Optional additional text to add before the final instruction
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    # Check if this is a predictive question type
    question_type = entry.get("metadata", {}).get("question_type", "")
    is_predictive = question_type == "predictive"
    
    # For predictive questions, use pre-cut video_partial.mp4
    video_filename = "video_partial.mp4" if is_predictive else "video.mp4"
    video_path = os.path.join(
        base_dir, "datasets", dataset_name, "shots", entry["video"], video_filename
    )
    
    if not os.path.exists(video_path):
        if is_predictive:
            raise FileNotFoundError(
                f"Pre-cut video not found: {video_path}. "
                f"Please run scripts/precut_test_videos.py to create video_partial.mp4 files."
            )
        else:
            raise FileNotFoundError(f"Video not found: {video_path}")
    
    question_prompt = build_question_prompt(entry, additional_suffix=additional_suffix)
    
    # Read and encode video
    with open(video_path, "rb") as video_file:
        video_b64 = base64.b64encode(video_file.read()).decode("utf-8")
    
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

