#!/usr/bin/env python3
"""
Batch Evaluation Script for Multiple Models

PURPOSE:
    This script simplifies submitting multiple evaluation jobs for different models.
    Instead of manually running sbatch multiple times, you can configure all models
    and their parameters in one place and submit them all at once.

USAGE:
    1. Edit the MODEL_CONFIGS dictionary below to specify which models to evaluate
       and their specific parameters
    2. Run: python scripts/batch_eval.py

    The script will submit one SLURM job per model, each with model-specific
    arguments plus any common defaults.

CONFIGURATION:
    Edit the MODEL_CONFIGS dictionary below. Each entry maps a model name to a
    dictionary of arguments. Arguments can include:
    
    Required:
        dataset: Dataset name (e.g., "ds1", "1k_simple")
    
    Optional (with defaults):
        num_samples: Number of samples per question (default: 1)
        max_concurrent: Maximum concurrent requests (default: 256)
        max_tokens: Maximum tokens for generation (default: None, uses model default)
        port: Port for vLLM server (default: 8000)
        vllm_sif: Path to vllm.sif file (default: None, uses auto-detection)
        max_entries: Limit to N entries (for quick testing, default: None)
        include_predictive: Include predictive.jsonl dataset (default: False)
        counterfactual_test_size: Limit counterfactual_test entries (default: None)
        descriptive_size: Limit descriptive entries (default: None)
        predictive_size: Limit predictive entries (default: None)

EXAMPLE WORKFLOW:

    1. Edit this script to configure your models:
       MODEL_CONFIGS = {
           "Qwen/Qwen3-VL-4B-Instruct": {
               "dataset": "ds1",
               "num_samples": 1,
               "max_concurrent": 256,
           },
           "Qwen/Qwen3-VL-4B-Thinking": {
               "dataset": "ds1",
               "num_samples": 1,
               "max_concurrent": 256,
               "max_tokens": 32768,  # Different max_tokens for this model
           },
       }

    2. Run the script:
       python scripts/batch_eval.py

    3. Monitor your jobs:
       squeue -u $USER

    4. View logs (the script will show you the exact commands):
       tail -f outputs/slurm/eval_<MODEL>_<JOB_ID>.out

OUTPUT:
    Each model gets its own SLURM job with:
    - Job name: eval_<MODEL_NAME> (sanitized)
    - Output:   outputs/slurm/eval_<MODEL>_<JOB_ID>.out
    - Error:    outputs/slurm/eval_<MODEL>_<JOB_ID>.err

    The script will print all job IDs and commands to monitor/logs after submission.

NOTES:
    - Each job uses 1 GPU and runs independently
    - Jobs can run in parallel if GPUs are available
    - If one job fails, others continue running
    - To cancel all jobs, use: scancel -u $USER (cancels all your jobs)
    - Model names must match presets in scripts/auto_eval.py
    - The script will wait for you to press Enter before submitting each job
      This gives you manual control over the timing between submissions
    
PORT AND GPU ASSIGNMENT:
    - Ports are automatically assigned starting from BASE_PORT (default: 8000)
      Each job gets a unique port: 8000, 8001, 8002, etc.
      This prevents port conflicts when multiple jobs run on the same node.
    - GPUs are automatically assigned by SLURM via --gpus-per-node=1
      SLURM handles GPU assignment across nodes and sets CUDA_VISIBLE_DEVICES
      The auto_eval.py script will auto-select an idle GPU based on nvidia-smi
      You control the delay between submissions to prevent conflicts
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# CONFIGURATION - Edit these to customize your batch evaluation
# ============================================================================

# Base port for vLLM servers (will be auto-incremented for each job)
# Each job will get a unique port: BASE_PORT, BASE_PORT+1, BASE_PORT+2, ...
BASE_PORT = 8000

# GPU assignment: SLURM automatically assigns GPUs via --gpus-per-node=1
# Each job requests 1 GPU, and SLURM sets CUDA_VISIBLE_DEVICES automatically
# The auto_eval.py script will auto-select an idle GPU based on nvidia-smi

# Default arguments applied to all models (can be overridden per-model)
# Note: 'port' will be auto-assigned if not specified per-model
COMMON_DEFAULTS = {
    "num_samples": 1,
    "max_concurrent": 256,
    # "port" will be auto-assigned if not specified
    "counterfactual_test_size": 384,
    "descriptive_size": 384,
    "predictive_size": 384,
    "vllm_sif": "~/scratch/vllm.sif",
    # Uncomment to set defaults for all models:
    # "max_tokens": 32768,
    # "max_entries": 10,
}

# Model-specific configurations
# Each model can have its own set of parameters
# Parameters not specified will use COMMON_DEFAULTS
# Note: If you specify 'port' explicitly, it will be used (but make sure it's unique!)
MODEL_CONFIGS: Dict[str, Dict[str, any]] = {
    # "Qwen/Qwen3-VL-4B-Instruct": {
    #     "dataset": "ds1",
    #     # "max_entries": 10,
    # },
    # "Qwen/Qwen3-VL-4B-Thinking": {
    #     "dataset": "ds1",
    #     "max_tokens": 32768,
    # },
    # "Qwen/Qwen3-VL-8B-Instruct": {
    #     "dataset": "ds1",
    # },
    "causalpool-4B": {
        "dataset": "ds1",
    },
}

# ============================================================================
# END CONFIGURATION
# ============================================================================


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in filenames and job names."""
    return re.sub(r'[\/\-]', '_', model_name)


def build_args(model: str, config: Dict[str, any]) -> List[str]:
    """
    Build the argument list for a model by merging COMMON_DEFAULTS and model config.
    
    Args:
        model: Model name
        config: Model-specific configuration dictionary
    
    Returns:
        List of argument strings in format ["--arg", "value", ...]
    """
    # Start with model name
    args = ["--model", model]
    
    # Merge defaults and model config (model config takes precedence)
    merged = {**COMMON_DEFAULTS, **config}
    
    # Map Python dict keys to command-line argument format
    arg_map = {
        "dataset": "--dataset",
        "num_samples": "--num-samples",
        "max_concurrent": "--max-concurrent",
        "max_tokens": "--max-tokens",
        "port": "--port",
        "vllm_sif": "--vllm-sif",
        "max_entries": "--max-entries",
        "include_predictive": "--include-predictive",  # Flag argument
        "counterfactual_test_size": "--counterfactual-test-size",
        "descriptive_size": "--descriptive-size",
        "predictive_size": "--predictive-size",
    }
    
    # Build argument list
    for key, value in merged.items():
        if value is None:
            continue  # Skip None values
        
        arg_name = arg_map.get(key)
        if arg_name is None:
            print(f"WARNING: Unknown parameter '{key}' for model '{model}', skipping", file=sys.stderr)
            continue
        
        # Handle flag arguments (boolean)
        if key == "include_predictive":
            if value:
                args.append(arg_name)
        else:
            # Regular argument with value
            args.extend([arg_name, str(value)])
    
    return args


def submit_job(
    model: str,
    config: Dict[str, any],
    project_dir: Path,
    port: int,
) -> Optional[int]:
    """
    Submit a SLURM job for a model evaluation.
    
    Args:
        model: Model name
        config: Model configuration dictionary
        project_dir: Project root directory
        port: Port number for vLLM server
    
    Returns:
        Job ID if successful, None otherwise
    
    Note:
        GPU assignment is handled automatically by SLURM via --gpus-per-node=1
        SLURM sets CUDA_VISIBLE_DEVICES automatically, which auto_eval.py respects
    """
    sanitized = sanitize_model_name(model)
    
    print(f"Submitting job for model: {model}")
    print(f"  Port: {port}")
    print(f"  GPU: auto-assigned by SLURM (--gpus-per-node=1)")
    
    # Build sbatch command
    # SLURM will automatically assign a GPU and set CUDA_VISIBLE_DEVICES
    sbatch_args = [
        "sbatch",
        "--job-name", f"eval_{sanitized}",
        "--output", f"outputs/slurm/eval_{sanitized}_%j.out",
        "--error", f"outputs/slurm/eval_{sanitized}_%j.err",
        "scripts/run_eval.sh",
    ]
    
    # Add model arguments (override port if not explicitly set in config)
    model_args = build_args(model, config)
    
    # Ensure port is set (override if not in config, or use config value)
    if "port" not in config or config.get("port") == COMMON_DEFAULTS.get("port", 8000):
        # Replace or add port argument
        port_set = False
        for i, arg in enumerate(model_args):
            if arg == "--port":
                model_args[i + 1] = str(port)
                port_set = True
                break
        if not port_set:
            model_args.extend(["--port", str(port)])
    else:
        # Port was explicitly set in config, use it but warn if it conflicts
        config_port = config.get("port", COMMON_DEFAULTS.get("port", 8000))
        if config_port != port:
            print(f"  WARNING: Model config specifies port {config_port}, but auto-assigned port {port} will be used", file=sys.stderr)
    
    sbatch_args.extend(model_args)
    
    # Submit job
    try:
        result = subprocess.run(
            sbatch_args,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Extract job ID from output
        output = result.stdout.strip()
        match = re.search(r'Submitted batch job (\d+)', output)
        if match:
            job_id = int(match.group(1))
            print(f"  ✓ Submitted job ID: {job_id}")
            return job_id
        else:
            print(f"  ✗ Failed to extract job ID from output: {output}", file=sys.stderr)
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to submit job: {e.stderr}", file=sys.stderr)
        return None


def main():
    """Main function to submit batch evaluation jobs."""
    # Change to project directory
    project_dir = Path("/home/scottc/links/scratch/causal_pool")
    os.chdir(project_dir)
    
    print("=" * 60)
    print("Batch Evaluation Submission")
    print("=" * 60)
    print(f"Number of models: {len(MODEL_CONFIGS)}")
    print(f"Common defaults: {COMMON_DEFAULTS}")
    print(f"Base port: {BASE_PORT} (ports will be {BASE_PORT} to {BASE_PORT + len(MODEL_CONFIGS) - 1})")
    print(f"GPU: Auto-assigned by SLURM (each job requests --gpus-per-node=1)")
    print(f"Submission: Press Enter to submit each job (you control the timing)")
    print()
    
    # Validate configurations
    for model, config in MODEL_CONFIGS.items():
        if "dataset" not in config:
            print(f"ERROR: Model '{model}' is missing required 'dataset' parameter", file=sys.stderr)
            sys.exit(1)
    
    # Submit jobs for each model with unique ports
    # GPUs are automatically assigned by SLURM
    job_ids = []
    for idx, (model, config) in enumerate(MODEL_CONFIGS.items()):
        # Assign unique port
        port = BASE_PORT + idx
        
        job_id = submit_job(model, config, project_dir, port)
        if job_id is not None:
            job_ids.append((model, job_id, port))
        else:
            print(f"ERROR: Failed to submit job for model '{model}'", file=sys.stderr)
            sys.exit(1)
        
        # Wait for user to press Enter before submitting the next job
        # This gives manual control over the timing between submissions
        if idx < len(MODEL_CONFIGS) - 1:  # Don't wait after the last job
            print()
            input(f"Press Enter to submit the next job ({idx + 2}/{len(MODEL_CONFIGS)})...")
            print()
        else:
            print()
    
    # Print summary
    print("=" * 60)
    print("Batch Submission Complete")
    print("=" * 60)
    print(f"Submitted {len(job_ids)} job(s):")
    for model, job_id, port in job_ids:
        print(f"  {model}: Job ID {job_id}, Port {port}, GPU (SLURM-assigned)")
    print()
    print("Monitor jobs with:")
    print("  squeue -u $USER")
    print()
    print("View logs:")
    for model, job_id, port in job_ids:
        sanitized = sanitize_model_name(model)
        print(f"  tail -f outputs/slurm/eval_{sanitized}_{job_id}.out")


if __name__ == "__main__":
    main()

