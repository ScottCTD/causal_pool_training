#!/usr/bin/env python3
"""
Manual interactive evaluation script for debugging.

This script loads a descriptive question from the test set, sends it to the model
with the same prompt format as eval.py, and then allows interactive conversation
with the model.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from openai import OpenAI
from causal_pool.eval.eval_utils import (
    build_prompt,
    get_model_hyperparameters,
    get_available_videos,
)
from causal_pool.data.dataset_utils import gather_test_dataset


def load_descriptive_question(dataset_name: str, base_dir: str = ".", index: int = 0) -> Dict[str, Any]:
    """
    Load a descriptive question from the test set.
    
    Args:
        dataset_name: Name of the dataset
        base_dir: Base directory for the project
        index: Index of the question to load (default: 0)
    
    Returns:
        Dataset entry dictionary
    """
    # Load all descriptive entries (-1 means all)
    sizes = {"descriptive": -1}
    
    # Change to base_dir for gather_test_dataset (it uses relative paths)
    original_cwd = os.getcwd()
    try:
        os.chdir(base_dir)
        dataset = gather_test_dataset(dataset_name, sizes, random_seed=42)
    finally:
        os.chdir(original_cwd)
    
    # Convert to list and get the entry
    entries = [entry for entry in dataset]
    if not entries:
        raise ValueError(f"No descriptive questions found in dataset {dataset_name}")
    
    print(f"Loaded {len(entries)} descriptive entries.")
    
    if index >= len(entries):
        raise ValueError(f"Index {index} out of range. Only {len(entries)} entries available (0-{len(entries)-1}).")
    
    return entries[index]


def print_entry_info(entry: Dict[str, Any]):
    """Print information about the dataset entry."""
    print("\n" + "=" * 70)
    print("DATASET ENTRY")
    print("=" * 70)
    print(f"Video: {entry.get('video', 'unknown')}")
    print(f"Question Type: {entry.get('metadata', {}).get('question_type', 'unknown')}")
    print(f"\nQuestion: {entry.get('question', 'N/A')}")
    print(f"\nOptions:")
    for i, option in enumerate(entry.get('options', [])):
        letter = chr(ord('A') + i)
        print(f"  {letter}. {option}")
    print(f"\nGround Truth: {entry.get('ground_truth', 'N/A')}")
    print("=" * 70 + "\n")


def extract_token_usage(response) -> Dict[str, Any]:
    """Extract token usage from API response."""
    token_usage = None
    if hasattr(response, 'usage') and response.usage is not None:
        token_usage = {
            'prompt_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else None,
            'completion_tokens': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else None,
            'total_tokens': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else None,
        }
    return token_usage


def print_token_usage(token_usage: Dict[str, Any]):
    """Print token usage information."""
    if token_usage:
        print("\nToken Usage:")
        if token_usage.get('prompt_tokens') is not None:
            print(f"  Input tokens:  {token_usage['prompt_tokens']:,}")
        if token_usage.get('completion_tokens') is not None:
            print(f"  Output tokens: {token_usage['completion_tokens']:,}")
        if token_usage.get('total_tokens') is not None:
            print(f"  Total tokens:  {token_usage['total_tokens']:,}")
    else:
        print("\nToken Usage: Not available")


def main():
    parser = argparse.ArgumentParser(
        description="Manual interactive evaluation script for debugging"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., '1k_simple')",
    )
    parser.add_argument(
        "-u", "--base-url",
        type=str,
        default="http://trig0002:8000/v1",
        help="Base URL for the API",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'Qwen/Qwen3-VL-4B-Instruct')",
    )
    parser.add_argument(
        "-k", "--api-key",
        type=str,
        default="EMPTY",
        help="API key (default: 'EMPTY')",
    )
    parser.add_argument(
        "-b", "--base-dir",
        type=str,
        default=".",
        help="Base directory for the project (default: current directory)",
    )
    parser.add_argument(
        "-i", "--index",
        type=int,
        default=0,
        help="Index of the descriptive question to load (0-based, default: 0). All descriptive entries are loaded.",
    )
    parser.add_argument(
        "-t", "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for generation (default: from model config)",
    )
    parser.add_argument(
        "-T", "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: from model config)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for video processing (default: 5)",
    )
    parser.add_argument(
        "--video-filename",
        type=str,
        default=None,
        help="Specific video filename to use (if None, uses first available)",
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Send only the video without the question text",
    )
    
    args = parser.parse_args()
    
    # Load descriptive question
    print(f"Loading descriptive question (index {args.index}) from dataset '{args.dataset}'...")
    try:
        entry = load_descriptive_question(args.dataset, args.base_dir, args.index)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Print entry information
    print_entry_info(entry)
    
    # Determine which video will be used
    try:
        available_videos = get_available_videos(entry, args.dataset, args.base_dir)
        if args.video_filename:
            if args.video_filename not in available_videos:
                print(f"Warning: Specified video '{args.video_filename}' not found in available videos.")
                print(f"Available videos: {available_videos}")
            video_filename = args.video_filename
        else:
            video_filename = available_videos[0] if available_videos else None
        
        # Construct full video path
        shot_dir = os.path.join(args.base_dir, "datasets", args.dataset, "shots", entry["video"])
        if video_filename:
            video_path = os.path.join(shot_dir, video_filename)
            print(f"\nVideo path: {video_path}")
            if available_videos:
                print(f"Available videos: {', '.join(available_videos)}")
        else:
            print(f"\nWarning: No video files found in {shot_dir}")
    except Exception as e:
        print(f"Warning: Could not determine video path: {e}")
        video_filename = args.video_filename
    
    # Build initial prompt (same as eval.py)
    print("\nBuilding prompt...")
    try:
        messages = build_prompt(
            entry,
            args.dataset,
            args.base_dir,
            video_filename=args.video_filename,
        )
        
        # If video-only mode, remove the text content
        if args.video_only:
            for message in messages:
                if message["role"] == "user" and "content" in message:
                    # Filter out text items, keep only video
                    message["content"] = [
                        item for item in message["content"]
                        if item.get("type") != "text"
                    ]
                    # Some APIs may require at least one text item, so add empty text
                    # Check if we have any text items left
                    has_text = any(item.get("type") == "text" for item in message["content"])
                    if not has_text and message["content"]:
                        # Add empty text item for API compatibility (if we have video content)
                        message["content"].append({"type": "text", "text": ""})
    except Exception as e:
        print(f"Error building prompt: {e}")
        sys.exit(1)
    
    # Print the text part of the prompt (without video data)
    print("\nInitial Prompt (text only):")
    print("-" * 70)
    text_found = False
    for message in messages:
        if message["role"] == "user":
            for content_item in message["content"]:
                if content_item.get("type") == "text":
                    text_content = content_item.get("text", "")
                    if text_content:
                        print(text_content)
                        text_found = True
                    break
    if not text_found:
        print("[Video only - no question text]")
    print("-" * 70 + "\n")
    
    # Get model hyperparameters
    model_hparams = get_model_hyperparameters(args.model)
    
    # Resolve hyperparameters
    max_tokens = args.max_tokens if args.max_tokens is not None else model_hparams.get("max_tokens")
    temperature = args.temperature if args.temperature is not None else model_hparams.get("temperature")
    
    # Build extra_body (same as AsyncEvaluator)
    extra_body_dict = {
        "mm_processor_kwargs": {"fps": args.fps, "do_sample_frames": True}
    }
    
    # Add keys from model_hparams if they exist and are not None
    for key in ["top_k", "top_p", "repetition_penalty", "presence_penalty"]:
        if key in model_hparams and model_hparams[key] is not None:
            extra_body_dict[key] = model_hparams[key]
    
    extra_body = extra_body_dict if extra_body_dict else None
    
    # Create OpenAI client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    
    print(f"Model: {args.model}")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    if extra_body:
        print(f"Extra body: {extra_body}")
    print("\nSending initial prompt to model...")
    print("=" * 70 + "\n")
    
    # Send initial prompt
    try:
        request_kwargs = {
            "messages": messages,
            "model": args.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        
        response = client.chat.completions.create(**request_kwargs)
        
        assistant_message = response.choices[0].message.content
        if assistant_message is None:
            print("Error: Empty response from model")
            sys.exit(1)
        
        # Extract token usage
        token_usage = extract_token_usage(response)
        
        print("ASSISTANT:")
        print(assistant_message)
        print_token_usage(token_usage)
        print("\n" + "=" * 70)
        
        # Add assistant response to conversation history
        messages.append({
            "role": "assistant",
            "content": assistant_message,
        })
        
    except Exception as e:
        print(f"Error calling API: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Interactive loop
    print("\nEntering interactive mode. Type your message and press Enter.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("Type 'clear' to clear conversation history and start over.")
    print("Type 'show' to show the current conversation history.")
    print("-" * 70 + "\n")
    
    while True:
        try:
            user_input = input("YOU: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break
            
            if user_input.lower() == 'clear':
                # Reset to initial prompt only
                messages = build_prompt(
                    entry,
                    args.dataset,
                    args.base_dir,
                    video_filename=args.video_filename,
                )
                # Apply video-only filter if enabled
                if args.video_only:
                    for message in messages:
                        if message["role"] == "user" and "content" in message:
                            message["content"] = [
                                item for item in message["content"]
                                if item.get("type") != "text"
                            ]
                            has_text = any(item.get("type") == "text" for item in message["content"])
                            if not has_text and message["content"]:
                                message["content"].append({"type": "text", "text": ""})
                print("\nConversation history cleared. Starting fresh with initial prompt.\n")
                continue
            
            if user_input.lower() == 'show':
                print("\n" + "=" * 70)
                print("CONVERSATION HISTORY")
                print("=" * 70)
                for i, msg in enumerate(messages):
                    role = msg["role"].upper()
                    content = msg["content"]
                    if isinstance(content, list):
                        # Extract text from content list
                        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                        content = "\n".join(text_parts) if text_parts else "[video + text]"
                    print(f"\n[{i+1}] {role}:")
                    print(content)
                print("=" * 70 + "\n")
                continue
            
            # Add user message to conversation
            messages.append({
                "role": "user",
                "content": user_input,
            })
            
            # Get response from model
            print("\nASSISTANT: ", end="", flush=True)
            request_kwargs = {
                "messages": messages,
                "model": args.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            
            response = client.chat.completions.create(**request_kwargs)
            
            assistant_message = response.choices[0].message.content
            if assistant_message is None:
                print("Error: Empty response from model")
                continue
            
            # Extract token usage
            token_usage = extract_token_usage(response)
            
            print(assistant_message)
            print_token_usage(token_usage)
            print()
            
            # Add assistant response to conversation history
            messages.append({
                "role": "assistant",
                "content": assistant_message,
            })
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue chatting.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

