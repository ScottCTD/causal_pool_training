#!/bin/bash
#SBATCH --job-name=auto_eval
#SBATCH --output=outputs/slurm/auto_eval_%j.out
#SBATCH --error=outputs/slurm/auto_eval_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-08:00:00
# Note: This requests exactly 1 GPU on 1 node for testing

# Automated evaluation script for Trillium cluster
#
# This script automates the full evaluation pipeline:
# 1. Loads required modules
# 2. Activates the virtual environment
# 3. Launches vLLM server for the specified model
# 4. Waits for server readiness
# 5. Runs eval.py against the local server
# 6. Cleans up the server process
#
# Usage:
#   sbatch scripts/run_eval.sbatch
#
# To customize the evaluation, modify the python command below with:
#   --model: Model preset name (e.g., "Qwen/Qwen3-VL-4B-Instruct")
#   --dataset: Dataset name (e.g., "1k_simple")
#   --num-samples: Number of samples per question (default: 1)
#   --max-concurrent: Maximum concurrent requests (default: 512)
#   --max-tokens: Maximum tokens for generation (optional, uses model default if not set)
#   --port: Port for vLLM server (default: 8000)
#   --vllm-sif: Path to vllm.sif file (default: vllm.sif in project root)
#
# Examples:
#   # Basic evaluation
#   python scripts/auto_eval.py --model "Qwen/Qwen3-VL-4B-Instruct" --dataset 1k_simple
#
#   # With custom parameters
#   python scripts/auto_eval.py \\
#     --model "Qwen/Qwen3-VL-4B-Thinking" \\
#     --dataset 1k_simple \\
#     --num-samples 1 \\
#     --max-concurrent 64 \\
#     --max-tokens 32768
#
#   # For causalpool-4B (merged checkpoint)
#   python scripts/auto_eval.py \\
#     --model "causalpool-4B" \\
#     --dataset 1k_simple \\
#     --vllm-sif ~/scratch/vllm.sif

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Change to project directory
PROJECT_DIR="/home/scottc/links/scratch/causal_pool"
cd "$PROJECT_DIR"

# Load .env file if it exists (export all variables automatically)
if [ -f .env ]; then
    set -a  # Automatically export all variables
    source .env
    set +a  # Turn off auto-export
fi

# Load modules
module load StdEnv/2023 gcc/12.3 cuda/12.6

# Set environment variables (these may override .env if needed)
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HUB_OFFLINE=1

# Activate virtual environment
source .venv/bin/activate

# Run automated evaluation
# Modify the arguments below to customize your evaluation
# The script will automatically:
# - Select an idle GPU
# - Launch vLLM server
# - Wait for server readiness
# - Run eval.py
# - Clean up and exit
#
# Additional arguments you can add for testing:
#   --max-entries N          : Limit to N entries (for quick testing)
#   --include-predictive     : Include predictive.jsonl dataset
#   --counterfactual-test-size N : Limit counterfactual_test entries
#   --descriptive-size N     : Limit descriptive entries
#   --predictive-size N      : Limit predictive entries
python scripts/auto_eval.py \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --dataset 1k_simple \
  --num-samples 1 \
  --max-concurrent 256 \
  --max-tokens 10 \
  --port 8000 \
#  --max-entries 10 \

# Job will terminate automatically after eval completes (or on error)

