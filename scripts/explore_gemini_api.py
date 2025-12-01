#!/usr/bin/env python3
"""
Call Gemini 3 Pro (preview/exp) with a video + prompt and custom FPS.

Usage:
  export GEMINI_API_KEY="your_api_key_from_ai_studio"
  python gemini3_video_fps.py \
      --video ./clip.mp4 \
      --prompt "Describe the main events in this clip with timestamps." \
      --fps 5.0 \
      --model models/gemini-3-pro-preview
"""

import argparse
import os
from google import genai
from google.genai import types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send video + text to Gemini 3 Pro with custom FPS."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (defaults to GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-pro",
        help=(
            "Model name. For Gemini 3.0 Pro experimental you may see "
            "IDs like 'models/gemini-3.0-pro-exp' in the docs or AI Studio. "
            "Override this if needed."
        ),
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file (ideally < 20 MB for inline upload).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to send along with the video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to sample via VideoMetadata.fps (default: 1.0).",
    )
    parser.add_argument(
        "--mime-type",
        type=str,
        default="video/mp4",
        help="MIME type of the video (default: video/mp4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing API key. Pass --api-key or set GEMINI_API_KEY env var."
        )

    # Read video bytes (inline upload; keep file < 20 MB)
    with open(args.video, "rb") as f:
        video_bytes = f.read()

    client = genai.Client(api_key=api_key)

    # Build the multimodal request:
    # - video as inline_data + video_metadata(fps=...)
    # - prompt as a text Part
    response = client.models.generate_content(
        model=args.model,
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(
                        data=video_bytes,
                        mime_type=args.mime_type,
                    ),
                    video_metadata=types.VideoMetadata(
                        fps=args.fps,
                    ),
                ),
                types.Part(text=args.prompt),
            ]
        ),
    )

    # Print the model's reply (text only)
    print(response.text)


if __name__ == "__main__":
    main()
