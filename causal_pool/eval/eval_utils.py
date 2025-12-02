"""
Utility functions for evaluation script.

This module contains:
- Model configuration utilities (hyperparameters, normalization)
- Evaluation utilities (metrics calculation, prediction validation)
- Prompt building utilities
"""

import base64
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from omegaconf import OmegaConf
from causal_pool.prompt_utils import build_question_prompt
from causal_pool.utils import normalize_model_name


class InvalidPredictionError(ValueError):
    """Raised when prediction format is invalid (not pure A-Z or has duplicates)."""
    pass


# Model name to eval config name mapping
MODEL_TO_EVAL_CONFIG: Dict[str, str] = {
    "Qwen/Qwen3-VL-4B-Instruct": "qwen_4b_instruct",
    "Qwen/Qwen3-VL-4B-Thinking": "qwen_4b_thinking",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen_8b_instruct",
    "CausalPool-4B-cf": "causalpool_4b_cf",
    "CausalPool-4B-desc": "causalpool_4b_desc",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen_30b_a3b_instruct",
    "Qwen/Qwen3-VL-32B-Instruct": "qwen_32b_instruct",
    "OpenGVLab/InternVL3_5-4B": "default",  # Use default if no specific config
}

# Default hyperparameters (used if model not found in configs)
DEFAULT_HYPERPARAMETERS = {
    "temperature": 0.8,
    "top_k": 20,
    "top_p": 0.8,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
}


def get_model_hyperparameters(model_name: str, base_dir: str = ".") -> Dict[str, Any]:
    """
    Get hyperparameters for a specific model from config files, falling back to defaults if not found.
    
    Args:
        model_name: Model name (e.g., "Qwen/Qwen3-VL-4B-Instruct")
        base_dir: Base directory for the project (default: current directory)
    
    Returns:
        Dictionary of hyperparameters
    """
    # Map model name to eval config name
    eval_config_name = MODEL_TO_EVAL_CONFIG.get(model_name, "default")
    
    # Load config files
    config_path = Path(base_dir) / "configs" / "eval" / "eval"
    default_file = config_path / "default.yaml"
    eval_file = config_path / f"{eval_config_name}.yaml"
    
    # Try to load from config files
    try:
        # Load default config first (always merge with default.yaml if it exists)
        default_cfg = OmegaConf.load(default_file) if default_file.exists() else OmegaConf.create({})
        # Remove defaults key if present (Hydra-specific)
        if "defaults" in default_cfg:
            del default_cfg["defaults"]
        
        # Load eval-specific config (will override defaults)
        eval_cfg = OmegaConf.load(eval_file) if eval_file.exists() else OmegaConf.create({})
        # Remove defaults key if present (Hydra-specific, we handle defaults by always merging with default.yaml)
        if "defaults" in eval_cfg:
            del eval_cfg["defaults"]
        
        # Merge configs
        cfg = OmegaConf.merge(default_cfg, eval_cfg)
        
        # Extract hyperparameters
        if "hyperparameters" in cfg:
            return dict(cfg.hyperparameters)
    except Exception:
        # If loading fails, fall back to defaults
        pass
    
    # Fall back to hardcoded defaults
    return DEFAULT_HYPERPARAMETERS.copy()


def get_available_videos(entry: Dict[str, Any], dataset_name: str, base_dir: str = ".") -> List[str]:
    """
    Get all available video files for a dataset entry.
    
    For predictive questions, returns all *-partial.mp4 files.
    For other questions, returns all *-full.mp4 files.
    
    Supports both new format (*-full.mp4, *-partial.mp4) and legacy format (video.mp4, video_partial.mp4).
    
    Args:
        entry: Dataset entry
        dataset_name: Name of the dataset
        base_dir: Base directory for the project
    
    Returns:
        List of video filenames (sorted for consistency)
    """
    # Check if this is a predictive question type
    question_type = entry.get("metadata", {}).get("question_type", "")
    is_predictive = question_type == "predictive"
    
    # Build shot directory path
    shot_dir = os.path.join(base_dir, "datasets", dataset_name, "shots", entry["video"])
    
    if not os.path.exists(shot_dir):
        raise FileNotFoundError(f"Shot directory not found: {shot_dir}")
    
    # Find all video files in the shot directory
    try:
        all_files = os.listdir(shot_dir)
    except Exception as e:
        raise FileNotFoundError(f"Error reading shot directory {shot_dir}: {e}")
    
    # For predictive questions, look for *-partial.mp4 files
    # For other questions, look for *-full.mp4 files
    if is_predictive:
        # Look for *-partial.mp4 files (new format) or video_partial.mp4 (legacy)
        video_files = [
            f for f in all_files 
            if f.endswith('-partial.mp4') or f == "video_partial.mp4"
        ]
        if not video_files:
            raise FileNotFoundError(
                f"No partial video files found in {shot_dir}. "
                f"Please run scripts/precut_test_videos.py to create *-partial.mp4 files."
            )
    else:
        # Look for *-full.mp4 files (new format) or video.mp4 (legacy)
        video_files = [
            f for f in all_files 
            if f.endswith('-full.mp4') or f == "video.mp4"
        ]
        if not video_files:
            raise FileNotFoundError(
                f"No full video files found in {shot_dir}. "
                f"Expected *-full.mp4 or video.mp4 files."
            )
    
    # Return sorted list for consistency
    return sorted(video_files)


def build_prompt(entry: Dict[str, Any], dataset_name: str, base_dir: str = ".", additional_suffix: str = None, video_filename: str = None) -> List[Dict[str, Any]]:
    """
    Build prompt for a dataset entry with a specific video file.
    
    For predictive questions, uses pre-cut *-partial.mp4 files instead of *-full.mp4.
    
    Supports both new format (*-full.mp4, *-partial.mp4) and legacy format (video.mp4, video_partial.mp4).
    
    Args:
        entry: Dataset entry
        dataset_name: Name of the dataset
        base_dir: Base directory for the project
        additional_suffix: Optional additional text to add before the final instruction
        video_filename: Specific video filename to use (if None, will select based on question type)
    
    Returns:
        List of message dictionaries for OpenAI API
    """
    # Check if this is a predictive question type
    question_type = entry.get("metadata", {}).get("question_type", "")
    is_predictive = question_type == "predictive"
    
    # Build shot directory path
    shot_dir = os.path.join(base_dir, "datasets", dataset_name, "shots", entry["video"])
    
    if not os.path.exists(shot_dir):
        raise FileNotFoundError(f"Shot directory not found: {shot_dir}")
    
    # If video_filename is provided, use it; otherwise determine based on question type
    if video_filename is None:
        available_videos = get_available_videos(entry, dataset_name, base_dir)
        if not available_videos:
            if is_predictive:
                raise FileNotFoundError(
                    f"No partial video files found in {shot_dir}. "
                    f"Please run scripts/precut_test_videos.py to create *-partial.mp4 files."
                )
            else:
                raise FileNotFoundError(
                    f"No full video files found in {shot_dir}. "
                    f"Expected *-full.mp4 or video.mp4 files."
                )
        video_filename = available_videos[0]  # Use first one as default
    
    video_path = os.path.join(shot_dir, video_filename)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
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

