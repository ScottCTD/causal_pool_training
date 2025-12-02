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
        counterfactual_velocity_size: Limit test-counterfactual_velocity entries (default: None)
        counterfactual_position_size: Limit test-counterfactual_position entries (default: None)
        descriptive_size: Limit test-descriptive entries (default: None)
        predictive_size: Limit test-predictive entries (default: None)

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
       tail -f outputs/slurm/eval_<MODEL>.out

OUTPUT:
    Each model gets its own SLURM job with:
    - Job name: eval_<MODEL_NAME> (sanitized)
    - Output:   outputs/slurm/eval_<MODEL>.out
    - Error:    outputs/slurm/eval_<MODEL>.err

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
      The auto_eval.py script uses all GPUs allocated by SLURM
      You control the delay between submissions to prevent conflicts
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in filenames and job names."""
    return re.sub(r'[\/\-]', '_', model_name)


def load_model_config(model_config_name: str, base_dir: str) -> DictConfig:
    """Load model configuration from YAML file."""
    config_path = Path(base_dir) / "configs" / "eval" / "models"
    model_file = config_path / f"{model_config_name}.yaml"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model config not found: {model_file}")
    
    return OmegaConf.load(model_file)


def load_eval_config(eval_config_name: str, base_dir: str) -> DictConfig:
    """Load evaluation configuration from YAML file."""
    config_path = Path(base_dir) / "configs" / "eval" / "eval"
    default_file = config_path / "default.yaml"
    eval_file = config_path / f"{eval_config_name}.yaml"
    
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
    return OmegaConf.merge(default_cfg, eval_cfg)


def build_args(model_cfg: DictConfig, eval_cfg: DictConfig, main_cfg: DictConfig) -> List[str]:
    """
    Build the argument list for a model by merging configs.
    
    Config precedence (highest to lowest):
    1. model_cfg.eval_overrides (model-specific overrides)
    2. main_cfg.dataset (global dataset setting)
    3. eval_cfg (model's eval config, e.g., eval/qwen_4b_instruct.yaml)
    4. eval/default.yaml (base eval config)
    
    Args:
        model_cfg: Model configuration (from configs/eval/models/)
        eval_cfg: Evaluation configuration (from configs/eval/eval/)
        main_cfg: Main configuration (from config.yaml) - contains global dataset setting
    
    Returns:
        List of argument strings in format ["--arg", "value", ...]
    """
    # Start with model name
    args = ["--model", model_cfg.model_name]
    
    # Merge configs in order: eval_cfg -> global dataset -> model overrides
    merged_eval = OmegaConf.merge(eval_cfg, {})
    
    # Add global dataset from main config if not in eval config
    if "dataset" in main_cfg and main_cfg.dataset:
        merged_eval["dataset"] = main_cfg.dataset
    
    # Apply model-specific overrides (highest priority)
    eval_overrides = model_cfg.get("eval_overrides") or {}
    merged_eval = OmegaConf.merge(merged_eval, eval_overrides)
    
    # Map config keys to command-line argument format
    arg_map = {
        "dataset": "--dataset",
        "num_samples": "--num-samples",
        "max_concurrent": "--max-concurrent",
        "max_tokens": "--max-tokens",
        "port": "--port",
        "vllm_sif": "--vllm-sif",
        "max_entries": "--max-entries",
        "include_predictive": "--include-predictive",  # Flag argument
        "counterfactual_velocity_size": "--counterfactual-velocity-size",
        "counterfactual_position_size": "--counterfactual-position-size",
        "descriptive_size": "--descriptive-size",
        "predictive_size": "--predictive-size",
    }
    
    # Build argument list from merged eval config
    for key, value in merged_eval.items():
        if value is None or key == "hyperparameters" or key == "base_url" or key == "api_key" or key == "fps":
            continue  # Skip None values and internal config keys
        
        arg_name = arg_map.get(key)
        if arg_name is None:
            continue  # Skip unknown keys
        
        # Handle flag arguments (boolean)
        if key == "include_predictive":
            if value:
                args.append(arg_name)
        else:
            # Regular argument with value
            args.extend([arg_name, str(value)])
    
    # Add vllm_sif from common config if not in eval config
    if "vllm_sif" not in merged_eval and "vllm_sif" in main_cfg.common:
        args.extend(["--vllm-sif", str(main_cfg.common.vllm_sif)])
    
    return args


def submit_job(
    model_cfg: DictConfig,
    eval_cfg: DictConfig,
    main_cfg: DictConfig,
    project_dir: Path,
    port: int,
) -> Optional[int]:
    """
    Submit a SLURM job for a model evaluation.
    
    Args:
        model_cfg: Model configuration
        eval_cfg: Evaluation configuration
        main_cfg: Main configuration (contains global settings like dataset)
        project_dir: Project root directory
        port: Port number for vLLM server
    
    Returns:
        Job ID if successful, None otherwise
    
    Note:
        GPU assignment is handled automatically by SLURM via --gpus-per-node=1
        SLURM sets CUDA_VISIBLE_DEVICES automatically, and auto_eval.py uses all allocated GPUs
    """
    model_name = model_cfg.model_name
    sanitized = sanitize_model_name(model_name)
    
    print(f"Submitting job for model: {model_name}")
    print(f"  Port: {port}")
    print(f"  GPU: allocated by SLURM (--gpus-per-node=1)")
    
    # Build sbatch command
    # SLURM will automatically assign a GPU and set CUDA_VISIBLE_DEVICES
    sbatch_args = [
        "sbatch",
        "--job-name", f"eval_{sanitized}",
        "--output", f"outputs/slurm/eval_{sanitized}.out",
        "--error", f"outputs/slurm/eval_{sanitized}.err",
        "scripts/run_eval.sh",
    ]
    
    # Add model arguments
    model_args = build_args(model_cfg, eval_cfg, main_cfg)
    
    # Ensure port is set
    port_set = False
    for i, arg in enumerate(model_args):
        if arg == "--port":
            model_args[i + 1] = str(port)
            port_set = True
            break
    if not port_set:
        model_args.extend(["--port", str(port)])
    
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


@hydra.main(version_base=None, config_path="../configs/eval", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to submit batch evaluation jobs."""
    # Change to project directory
    project_dir = Path(cfg.common.base_dir)
    os.chdir(project_dir)
    
    # Get models to evaluate
    models_to_evaluate = cfg.models_to_evaluate
    base_port = cfg.common.base_port
    
    print("=" * 60)
    print("Batch Evaluation Submission")
    print("=" * 60)
    print(f"Number of models: {len(models_to_evaluate)}")
    print(f"Models: {', '.join(models_to_evaluate)}")
    print(f"Base port: {base_port} (ports will be {base_port} to {base_port + len(models_to_evaluate) - 1})")
    print(f"GPU: Allocated by SLURM (each job requests --gpus-per-node=1)")
    print(f"Submission: Press Enter to submit each job (you control the timing)")
    print()
    
    # Load configs for each model
    model_configs = []
    for model_config_name in models_to_evaluate:
        try:
            model_cfg = load_model_config(model_config_name, project_dir)
            eval_config_name = model_cfg.get("eval_config", "default")
            eval_cfg = load_eval_config(eval_config_name, project_dir)
            model_configs.append((model_config_name, model_cfg, eval_cfg))
        except Exception as e:
            print(f"ERROR: Failed to load config for model '{model_config_name}': {e}", file=sys.stderr)
            sys.exit(1)
    
    # Validate configurations
    if "dataset" not in cfg or cfg.dataset is None:
        print(f"ERROR: Global 'dataset' parameter is missing in config.yaml", file=sys.stderr)
        sys.exit(1)
    
    # Submit jobs for each model with unique ports
    # GPUs are automatically assigned by SLURM
    job_ids = []
    for idx, (model_config_name, model_cfg, eval_cfg) in enumerate(model_configs):
        # Assign unique port
        port = base_port + idx
        
        job_id = submit_job(model_cfg, eval_cfg, cfg, project_dir, port)
        if job_id is not None:
            job_ids.append((model_cfg.model_name, job_id, port))
        else:
            print(f"ERROR: Failed to submit job for model '{model_config_name}'", file=sys.stderr)
            sys.exit(1)
        
        # Wait for user to press Enter before submitting the next job
        # This gives manual control over the timing between submissions
        if idx < len(model_configs) - 1:  # Don't wait after the last job
            print()
            input(f"Press Enter to submit the next job ({idx + 2}/{len(model_configs)})...")
            print()
        else:
            print()
    
    # Print summary
    print("=" * 60)
    print("Batch Submission Complete")
    print("=" * 60)
    print(f"Submitted {len(job_ids)} job(s):")
    for model, job_id, port in job_ids:
        print(f"  {model}: Job ID {job_id}, Port {port}, GPU (SLURM-allocated)")
    print()
    print("Monitor jobs with:")
    print("  squeue -u $USER")
    print()
    print("View logs:")
    for model, job_id, port in job_ids:
        sanitized = sanitize_model_name(model)
        print(f"  tail -f outputs/slurm/eval_{sanitized}.out")


if __name__ == "__main__":
    main()

