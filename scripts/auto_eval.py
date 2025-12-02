#!/usr/bin/env python3
"""
Automated evaluation orchestrator for Trillium cluster.

This script:
1. Launches vLLM server via Apptainer for the specified model
2. Waits for the server to be ready
3. Runs eval.py against the local server
4. Cleans up the server process

GPU assignment is handled by SLURM. The script uses all GPUs allocated by SLURM
by not setting CUDA_VISIBLE_DEVICES.
"""

# Set thread limits before importing any libraries that might spawn threads
# This prevents excessive thread creation on login nodes
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf


def load_vllm_config(model_name: str, base_dir: str) -> DictConfig:
    """
    Load vLLM config for a model from YAML files.
    
    Maps model names to config names:
    - "Qwen/Qwen3-VL-4B-Instruct" -> "qwen_4b_instruct"
    - "Qwen/Qwen3-VL-4B-Thinking" -> "qwen_4b_thinking"
    - "Qwen/Qwen3-VL-8B-Instruct" -> "qwen_8b_instruct"
    - "CausalPool-4B-cf" -> "causalpool_4b_cf"
    - "CausalPool-4B-desc" -> "causalpool_4b_desc"
    - "Qwen/Qwen3-VL-30B-A3B-Instruct" -> "qwen_30b_a3b_instruct"
    - "Qwen/Qwen3-VL-32B-Instruct" -> "qwen_32b_instruct"
    - "OpenGVLab/InternVL3_5-4B" -> "internvl3_5_4b"
    """
    # Map model names to config names
    model_to_config = {
        "Qwen/Qwen3-VL-4B-Instruct": "qwen_4b_instruct",
        "Qwen/Qwen3-VL-4B-Thinking": "qwen_4b_thinking",
        "Qwen/Qwen3-VL-8B-Instruct": "qwen_8b_instruct",
        "CausalPool-4B-cf": "causalpool_4b_cf",
        "CausalPool-4B-desc": "causalpool_4b_desc",
        "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen_30b_a3b_instruct",
        "Qwen/Qwen3-VL-32B-Instruct": "qwen_32b_instruct",
        "OpenGVLab/InternVL3_5-4B": "internvl3_5_4b",
    }
    
    config_name = model_to_config.get(model_name)
    if config_name is None:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_to_config.keys())}")
    
    # Load config files
    config_path = Path(base_dir) / "configs" / "eval" / "vllm"
    default_file = config_path / "default.yaml"
    model_file = config_path / f"{config_name}.yaml"
    
    if not model_file.exists():
        raise FileNotFoundError(f"vLLM config not found: {model_file}")
    
    # Load default config first (always merge with default.yaml if it exists)
    default_cfg = OmegaConf.load(default_file) if default_file.exists() else OmegaConf.create({})
    # Remove defaults key if present (Hydra-specific, not needed for OmegaConf)
    if "defaults" in default_cfg:
        del default_cfg["defaults"]
    
    # Load model-specific config (will override defaults)
    model_cfg = OmegaConf.load(model_file)
    # Remove defaults key if present (Hydra-specific, we handle defaults by always merging with default.yaml)
    if "defaults" in model_cfg:
        del model_cfg["defaults"]
    
    # Merge configs (model config overrides defaults)
    cfg = OmegaConf.merge(default_cfg, model_cfg)
    
    return cfg


def log(message: str, prefix: str = "[AUTO-EVAL]"):
    """Print a log message with prefix."""
    print(f"{prefix} {message}", flush=True)


def wait_for_server(base_url: str, timeout: int = 600, interval: int = 5) -> bool:
    """
    Wait for the vLLM server to become ready.
    
    Args:
        base_url: Base URL of the server (e.g., "http://127.0.0.1:8000/v1")
        timeout: Maximum time to wait in seconds (default: 600 = 10 minutes)
        interval: Time between checks in seconds (default: 5)
    
    Returns:
        True if server is ready, False if timeout
    """
    models_url = f"{base_url}/models"
    start_time = time.time()
    attempt = 0
    
    log(f"Waiting for server to be ready at {models_url} (timeout: {timeout}s)...")
    
    while time.time() - start_time < timeout:
        attempt += 1
        try:
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    elapsed = time.time() - start_time
                    log(f"Server is ready! (took {elapsed:.1f}s, {attempt} attempts)")
                    return True
        except urllib.error.URLError:
            # Server not ready yet, continue waiting
            pass
        except Exception as e:
            log(f"Unexpected error checking server (attempt {attempt}): {e}", prefix="[AUTO-EVAL] [WARN]")
        
        if attempt % 6 == 0:  # Log every 30 seconds (6 * 5s interval)
            elapsed = time.time() - start_time
            log(f"Still waiting... ({elapsed:.1f}s elapsed, {attempt} attempts)")
        
        time.sleep(interval)
    
    elapsed = time.time() - start_time
    log(f"ERROR: Server did not become ready within {timeout}s ({elapsed:.1f}s elapsed)", prefix="[AUTO-EVAL] [ERROR]")
    return False


def build_vllm_command(
    model_name: str,
    port: int,
    vllm_sif_path: str,
    base_dir: str,
) -> List[str]:
    """
    Build the Apptainer command to launch vLLM server.
    
    Args:
        model_name: Model preset name
        port: Port number for the server
        vllm_sif_path: Path to vllm.sif file
        base_dir: Base directory for the project
    
    Returns:
        List of command arguments for subprocess
    """
    # Load vLLM config
    vllm_cfg = load_vllm_config(model_name, base_dir)
    
    # Build vLLM args from config
    vllm_args = []
    
    # Model name (use model_path if available, otherwise model)
    if vllm_cfg.get("model_path"):
        model_path = vllm_cfg.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(base_dir, model_path)
        vllm_args.extend(["--model", model_path])
        if vllm_cfg.get("served_model_name"):
            vllm_args.extend(["--served-model-name", vllm_cfg.served_model_name])
    else:
        vllm_args.extend(["--model", vllm_cfg.model])
    
    # Required args
    vllm_args.extend(["--host", str(vllm_cfg.host)])
    vllm_args.extend(["--port", str(port)])
    
    # Optional args
    if vllm_cfg.get("tensor_parallel_size"):
        vllm_args.extend(["--tensor-parallel-size", str(vllm_cfg.tensor_parallel_size)])
    if vllm_cfg.get("gpu_memory_utilization"):
        vllm_args.extend(["--gpu-memory-utilization", str(vllm_cfg.gpu_memory_utilization)])
    if vllm_cfg.get("max_model_len"):
        vllm_args.extend(["--max-model-len", str(vllm_cfg.max_model_len)])
    if vllm_cfg.get("max_num_seqs"):
        vllm_args.extend(["--max-num-seqs", str(vllm_cfg.max_num_seqs)])
    if vllm_cfg.get("enforce_eager", False):
        vllm_args.append("--enforce-eager")
    if vllm_cfg.get("trust_remote_code", False):
        vllm_args.append("--trust-remote-code")
    if vllm_cfg.get("reasoning_parser"):
        vllm_args.extend(["--reasoning-parser", str(vllm_cfg.reasoning_parser)])
    if vllm_cfg.get("enable_expert_parallel", False):
        vllm_args.append("--enable-expert-parallel")
    
    # Handle media_io_kwargs (nested argument)
    if vllm_cfg.get("media_io_kwargs"):
        media_io = vllm_cfg.media_io_kwargs
        if media_io.get("video") and media_io.video.get("num_frames") is not None:
            vllm_args.extend(["--media-io-kwargs.video.num_frames", str(media_io.video.num_frames)])
    
    # Build Apptainer command
    cmd = [
        "apptainer", "run", "--nv",
        "--bind", "/scratch/scottc/:/scratch/scottc/",
        "--bind", "/home/scottc/links/:/home/scottc/links/",
        "--bind", "/scratch/scottc/cache/:/home/scottc/.cache",
        "--bind", "/scratch/scottc/cache/triton/:/home/scottc/.triton/",
        "--env", "HF_HUB_OFFLINE=1",
        "--env", "VLLM_NO_USAGE_STATS=1",
        vllm_sif_path,
    ] + vllm_args
    
    return cmd


def check_prerequisites(vllm_sif_path: str, base_dir: str) -> Tuple[bool, Optional[str]]:
    """
    Check if prerequisites are met.
    
    Returns:
        (success, error_message)
    """
    # Check vllm.sif exists
    if not os.path.exists(vllm_sif_path):
        return False, f"vLLM SIF file not found: {vllm_sif_path}"
    
    # Check bind directories exist
    bind_dirs = [
        "/scratch/scottc/",
        "/home/scottc/links/",
        "/scratch/scottc/cache/",
    ]
    for bind_dir in bind_dirs:
        if not os.path.exists(bind_dir):
            return False, f"Bind directory does not exist: {bind_dir}"
    
    # Check that causal_pool package exists (for module import)
    causal_pool_path = os.path.join(base_dir, "causal_pool")
    if not os.path.exists(causal_pool_path):
        return False, f"causal_pool package not found: {causal_pool_path}"
    
    # Check that eval module exists
    eval_module_path = os.path.join(causal_pool_path, "eval", "eval.py")
    if not os.path.exists(eval_module_path):
        return False, f"eval.py module not found: {eval_module_path}"
    
    return True, None


def main():
    parser = argparse.ArgumentParser(
        description="Automated evaluation orchestrator for Trillium cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default settings (via SLURM)
  sbatch scripts/run_eval.sh --model "Qwen/Qwen3-VL-4B-Instruct" --dataset ds1
  
  # Run directly on compute node (when already on a compute node)
  bash scripts/run_eval_direct.sh --model "Qwen/Qwen3-VL-4B-Instruct" --dataset ds1
  
  # Or set up environment manually and run:
  source scripts/setup_eval_env.sh
  python scripts/auto_eval.py --model "Qwen/Qwen3-VL-4B-Instruct" --dataset ds1
  
  # Run with custom parameters
  bash scripts/run_eval_direct.sh \\
    --model "Qwen/Qwen3-VL-4B-Thinking" \\
    --dataset ds1 \\
    --num-samples 1 \\
    --max-concurrent 64 \\
    --max-tokens 32768
        """,
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model name (must match a preset key, e.g., 'Qwen/Qwen3-VL-4B-Instruct')",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., '1k_simple')",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=1,
        help="Number of samples per question (default: 1)",
    )
    parser.add_argument(
        "-c", "--max-concurrent",
        type=int,
        default=256,
        help="Maximum concurrent requests (default: 512)",
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
        "-p", "--port",
        type=int,
        default=8000,
        help="Port for vLLM server (default: 8000)",
    )
    parser.add_argument(
        "-b", "--base-dir",
        type=str,
        default=".",
        help="Base directory for the project (default: current directory)",
    )
    parser.add_argument(
        "--vllm-sif",
        type=str,
        default="vllm.sif",
        help="Path to vllm.sif file (default: vllm.sif in current directory)",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        help="Timeout for server readiness in seconds (default: 600)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    # Pass-through arguments for eval.py
    parser.add_argument(
        "-e", "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to evaluate (for testing)",
    )
    parser.add_argument(
        "--include-predictive",
        action="store_true",
        help="Include predictive.jsonl dataset",
    )
    parser.add_argument(
        "-C", "--counterfactual-velocity-size",
        type=int,
        default=None,
        help="Number of entries to load from test-counterfactual_velocity.jsonl",
    )
    parser.add_argument(
        "--counterfactual-position-size",
        type=int,
        default=None,
        help="Number of entries to load from test-counterfactual_position.jsonl",
    )
    parser.add_argument(
        "-D", "--descriptive-size",
        type=int,
        default=None,
        help="Number of entries to load from test-descriptive.jsonl",
    )
    parser.add_argument(
        "-P", "--predictive-size",
        type=int,
        default=None,
        help="Number of entries to load from test-predictive.jsonl",
    )
    
    args = parser.parse_args()
    
    # Resolve base_dir to absolute path
    base_dir = os.path.abspath(args.base_dir)
    
    # Resolve vllm_sif_path
    if os.path.isabs(args.vllm_sif):
        vllm_sif_path = os.path.expanduser(args.vllm_sif)
    else:
        # Try base_dir first, then current directory, then home/scratch
        vllm_sif_path = os.path.join(base_dir, args.vllm_sif)
        if not os.path.exists(vllm_sif_path):
            vllm_sif_path = os.path.join(os.getcwd(), args.vllm_sif)
        if not os.path.exists(vllm_sif_path):
            # Try ~/scratch/vllm.sif for causalpool-4B case
            home_scratch_path = os.path.expanduser("~/scratch/vllm.sif")
            if os.path.exists(home_scratch_path):
                vllm_sif_path = home_scratch_path
    
    log("=" * 60)
    log("Starting automated evaluation")
    log("=" * 60)
    log(f"Model: {args.model}")
    log(f"Dataset: {args.dataset}")
    log(f"Base directory: {base_dir}")
    log(f"vLLM SIF path: {vllm_sif_path}")
    
    # Validate model preset (try loading config)
    try:
        load_vllm_config(args.model, base_dir)
    except (ValueError, FileNotFoundError) as e:
        log(f"ERROR: {e}", prefix="[AUTO-EVAL] [ERROR]")
        sys.exit(1)
    
    # Check prerequisites
    log("Checking prerequisites...")
    success, error_msg = check_prerequisites(vllm_sif_path, base_dir)
    if not success:
        log(f"ERROR: {error_msg}", prefix="[AUTO-EVAL] [ERROR]")
        sys.exit(1)
    log("Prerequisites check passed")
    
    # GPU assignment: SLURM allocates GPUs and sets CUDA_VISIBLE_DEVICES automatically
    # We use all allocated GPUs by not overriding CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        log(f"Using GPUs allocated by SLURM: CUDA_VISIBLE_DEVICES={cuda_visible}")
    else:
        log("No CUDA_VISIBLE_DEVICES set - will use all available GPUs")
    
    # Build vLLM command
    log("Building vLLM server command...")
    vllm_cmd = build_vllm_command(args.model, args.port, vllm_sif_path, base_dir)
    log(f"vLLM command: {' '.join(vllm_cmd)}")
    
    if args.dry_run:
        log("DRY RUN: Would execute vLLM command above")
        log("DRY RUN: Would wait for server readiness")
        log("DRY RUN: Would run eval.py")
        return
    
    # Launch vLLM server
    log("Launching vLLM server...")
    server_process = None
    eval_result = None
    exit_code = 0  # Default exit code
    server_log_file = None
    try:
        # Redirect server output to a log file to avoid pipe buffer blocking
        # If we use PIPE without reading, the buffer fills up and blocks the server
        server_log_path = os.path.join(
            base_dir,
            "outputs",
            "slurm",
            f"vllm_server_{args.model.replace('/', '_').replace('-', '_')}_{args.port}.log"
        )
        os.makedirs(os.path.dirname(server_log_path), exist_ok=True)
        server_log_file = open(server_log_path, "w")
        log(f"Server logs will be written to: {server_log_path}")
        
        server_process = subprocess.Popen(
            vllm_cmd,
            stdout=server_log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        log(f"vLLM server process started (PID: {server_process.pid})")
        
        # Wait for server to be ready
        base_url = f"http://127.0.0.1:{args.port}/v1"
        if not wait_for_server(base_url, timeout=args.server_timeout):
            log("ERROR: Server failed to become ready", prefix="[AUTO-EVAL] [ERROR]")
            if server_process:
                log("Terminating server process...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log("Force killing server process...")
                    server_process.kill()
                    server_process.wait()
            sys.exit(1)
        
        # Run eval.py
        log("=" * 60)
        log("Running evaluation")
        log("=" * 60)
        
        # Run eval.py as a module to ensure proper Python path resolution
        # This allows causal_pool imports to work correctly
        eval_cmd = [
            sys.executable, "-m", "causal_pool.eval.eval",
            "--dataset", args.dataset,
            "--base-url", base_url,
            "--model", args.model,
            "--num-samples", str(args.num_samples),
            "--max-concurrent", str(args.max_concurrent),
            "--base-dir", base_dir,
        ]
        
        if args.max_tokens is not None:
            eval_cmd.extend(["--max-tokens", str(args.max_tokens)])
        if args.temperature is not None:
            eval_cmd.extend(["--temperature", str(args.temperature)])
        if args.max_entries is not None:
            eval_cmd.extend(["--max-entries", str(args.max_entries)])
        if args.include_predictive:
            eval_cmd.append("--include-predictive")
        if args.counterfactual_velocity_size is not None:
            eval_cmd.extend(["--counterfactual-velocity-size", str(args.counterfactual_velocity_size)])
        if args.counterfactual_position_size is not None:
            eval_cmd.extend(["--counterfactual-position-size", str(args.counterfactual_position_size)])
        if args.descriptive_size is not None:
            eval_cmd.extend(["--descriptive-size", str(args.descriptive_size)])
        if args.predictive_size is not None:
            eval_cmd.extend(["--predictive-size", str(args.predictive_size)])
        
        log(f"Eval command: {' '.join(eval_cmd)}")
        
        # Change to base_dir and set PYTHONPATH for eval.py module import
        original_cwd = os.getcwd()
        original_pythonpath = os.environ.get("PYTHONPATH", "")
        try:
            os.chdir(base_dir)
            log("Changed working directory to base_dir")
            
            # Ensure base_dir is in PYTHONPATH for module imports
            if original_pythonpath:
                os.environ["PYTHONPATH"] = f"{base_dir}:{original_pythonpath}"
            else:
                os.environ["PYTHONPATH"] = base_dir
            
            eval_start_time = time.time()
            eval_result = subprocess.run(eval_cmd, cwd=base_dir, env=os.environ.copy())
            eval_elapsed = time.time() - eval_start_time
            
            if eval_result.returncode == 0:
                log(f"Evaluation completed successfully (took {eval_elapsed:.1f}s)")
            else:
                log(f"ERROR: Evaluation failed with return code {eval_result.returncode}", prefix="[AUTO-EVAL] [ERROR]")
                log(f"Evaluation took {eval_elapsed:.1f}s", prefix="[AUTO-EVAL] [ERROR]")
        finally:
            os.chdir(original_cwd)
            # Restore original PYTHONPATH
            if original_pythonpath:
                os.environ["PYTHONPATH"] = original_pythonpath
            elif "PYTHONPATH" in os.environ:
                del os.environ["PYTHONPATH"]
        
    except KeyboardInterrupt:
        log("\nInterrupted by user", prefix="[AUTO-EVAL] [WARN]")
        exit_code = 130  # Standard exit code for SIGINT
    except Exception as e:
        log(f"ERROR: Unexpected error: {e}", prefix="[AUTO-EVAL] [ERROR]")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # Clean up server process
        if server_process:
            log("=" * 60)
            log("Cleaning up vLLM server...")
            log("=" * 60)
            try:
                log("Sending TERM signal to server process...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=30)
                    log("Server process terminated gracefully")
                except subprocess.TimeoutExpired:
                    log("Server did not terminate within 30s, sending KILL signal...")
                    server_process.kill()
                    server_process.wait()
                    log("Server process killed")
            except Exception as e:
                log(f"WARNING: Error during cleanup: {e}", prefix="[AUTO-EVAL] [WARN]")
        
        # Close server log file
        if server_log_file:
            try:
                server_log_file.close()
            except Exception as e:
                log(f"WARNING: Error closing server log file: {e}", prefix="[AUTO-EVAL] [WARN]")
    
    log("=" * 60)
    log("Automated evaluation finished")
    log("=" * 60)
    
    # Determine exit code: prioritize eval result, then exception code, then error
    if eval_result is not None:
        exit_code = eval_result.returncode
    # If exit_code was set by exception handler, it's already correct
    # If neither eval_result nor exception occurred, exit_code remains 0 (shouldn't happen)
    
    log(f"Exiting with code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

