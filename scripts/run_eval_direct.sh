#!/bin/bash
# Direct evaluation script for running on compute nodes without SLURM
#
# USAGE:
#   When already on a compute node (e.g., via srun or interactive session):
#     bash scripts/run_eval_direct.sh --model "MODEL_NAME" --dataset DATASET_NAME [OPTIONS]
#
#   Or activate the environment manually and run:
#     source scripts/setup_eval_env.sh
#     python scripts/auto_eval.py --model "MODEL_NAME" --dataset DATASET_NAME [OPTIONS]
#
# REQUIRED ARGUMENTS:
#   --model MODEL_NAME     Model preset name (must match a preset in auto_eval.py)
#   --dataset DATASET_NAME Dataset name to evaluate
#
# OPTIONAL ARGUMENTS:
#   All arguments are passed through to scripts/auto_eval.py. Common options:
#   --num-samples N        Number of samples per question (default: 1)
#   --max-concurrent N     Maximum concurrent requests (default: 256)
#   --max-tokens N         Maximum tokens for generation (optional, uses model default)
#   --port N               Port for vLLM server (default: 8000)
#   --vllm-sif PATH        Path to vllm.sif file (default: vllm.sif in project root)
#
# EXAMPLES:
#   # Basic evaluation
#   bash scripts/run_eval_direct.sh \
#     --model "Qwen/Qwen3-VL-4B-Instruct" \
#     --dataset ds1
#
#   # With custom parameters
#   bash scripts/run_eval_direct.sh \
#     --model "Qwen/Qwen3-VL-4B-Thinking" \
#     --dataset ds1 \
#     --num-samples 1 \
#     --max-concurrent 256 \
#     --max-tokens 32768
#
# NOTES:
#   - This script assumes you're already on a compute node with GPU access
#   - GPU assignment: Uses all available GPUs (or set CUDA_VISIBLE_DEVICES manually)
#   - For SLURM jobs, use scripts/run_eval.sh instead

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

# Load modules (if not already loaded)
if ! command -v python3 &> /dev/null || ! python3 -c "import torch" &> /dev/null 2>&1; then
    echo "Loading modules..."
    module load StdEnv/2023 gcc/12.3 cuda/12.6 2>/dev/null || echo "Warning: Could not load modules (may already be loaded)"
fi

# Set environment variables (these may override .env if needed)
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HUB_OFFLINE=1
# Limit thread spawning for libraries
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Activate virtual environment
if [ -d .venv ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment .venv not found. Please create it first."
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -1
    echo ""
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Run automated evaluation
# All arguments passed to this script are forwarded to auto_eval.py
echo "Running auto_eval.py with arguments: $@"
python scripts/auto_eval.py "$@"

