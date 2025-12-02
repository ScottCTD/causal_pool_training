#!/bin/bash
# Setup script for evaluation environment
# Source this file to set up the environment, then run auto_eval.py directly
#
# USAGE:
#   source scripts/setup_eval_env.sh
#   python scripts/auto_eval.py --model "MODEL_NAME" --dataset DATASET_NAME [OPTIONS]

# Change to project directory
PROJECT_DIR="/home/scottc/links/scratch/causal_pool"
cd "$PROJECT_DIR"

# Load .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Load modules (if not already loaded)
if ! command -v python3 &> /dev/null || ! python3 -c "import torch" &> /dev/null 2>&1; then
    echo "Loading modules..."
    module load StdEnv/2023 gcc/12.3 cuda/12.6 2>/dev/null || echo "Warning: Could not load modules"
fi

# Set environment variables
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HUB_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Activate virtual environment
if [ -d .venv ]; then
    source .venv/bin/activate
    echo "Environment activated. You can now run:"
    echo "  python scripts/auto_eval.py --model \"MODEL_NAME\" --dataset DATASET_NAME [OPTIONS]"
else
    echo "ERROR: Virtual environment .venv not found."
    return 1 2>/dev/null || exit 1
fi

