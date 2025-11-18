"""
Video processing utilities for pre-processing videos.

This module contains utilities for cutting videos, used by pre-processing scripts
like scripts/precut_test_videos.py.
"""

import subprocess


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


def cut_video_fraction(video_path: str, output_path: str, fraction: float = 0.5) -> None:
    """
    Cut video to first fraction using ffmpeg.
    
    First tries codec copy (fast), falls back to re-encoding if that fails.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save cut video
        fraction: Fraction of video to retain (e.g., 0.3 for first 30%, default: 0.5)
    """
    if not 0 < fraction <= 1.0:
        raise ValueError(f"Fraction must be between 0 and 1, got {fraction}")
    
    duration = get_video_duration(video_path)
    cut_duration = duration * fraction
    
    # First try codec copy (fast, no re-encoding)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-t", str(cut_duration),
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
                "-t", str(cut_duration),
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


def cut_video_first_half(video_path: str, output_path: str) -> None:
    """
    Cut video to first half using ffmpeg.
    
    Convenience wrapper around cut_video_fraction with fraction=0.5.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save cut video
    """
    cut_video_fraction(video_path, output_path, fraction=0.5)


