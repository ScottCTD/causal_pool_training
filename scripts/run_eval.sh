#!/bin/bash
#SBATCH --job-name=auto_eval
#SBATCH --output=outputs/slurm/auto_eval_%j.out
#SBATCH --error=outputs/slurm/auto_eval_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-08:00:00
# Note: This requests exactly 1 GPU on 1 node for testing

################################################################################
# SLURM Evaluation Script for Trillium Cluster
################################################################################
#
# PURPOSE:
#   This script automates the full evaluation pipeline for a single model:
#   1. Loads required modules (StdEnv/2023, gcc/12.3, cuda/12.6)
#   2. Activates the virtual environment (.venv)
#   3. Launches vLLM server for the specified model via Apptainer
#   4. Waits for server readiness
#   5. Runs eval.py against the local server
#   6. Cleans up the server process
#
# USAGE:
#   Submit a single evaluation job:
#     sbatch scripts/run_eval.sh --model "MODEL_NAME" --dataset DATASET_NAME [OPTIONS]
#
#   For multiple models, use batch_eval.py instead:
#     python scripts/batch_eval.py
#
# REQUIRED ARGUMENTS:
#   --model MODEL_NAME     Model preset name (must match a preset in auto_eval.py)
#                         Examples: "Qwen/Qwen3-VL-4B-Instruct"
#                                   "Qwen/Qwen3-VL-4B-Thinking"
#                                   "Qwen/Qwen3-VL-8B-Instruct"
#                                   "causalpool-4B"
#
#   --dataset DATASET_NAME Dataset name to evaluate
#                         Examples: "ds1", "1k_simple"
#
# OPTIONAL ARGUMENTS:
#   All arguments are passed through to scripts/auto_eval.py. Common options:
#
#   --num-samples N        Number of samples per question (default: 1)
#   --max-concurrent N     Maximum concurrent requests (default: 256)
#   --max-tokens N         Maximum tokens for generation (optional, uses model default)
#   --port N               Port for vLLM server (default: 8000)
#   --vllm-sif PATH        Path to vllm.sif file (default: vllm.sif in project root)
#
#   Testing/debugging options:
#   --max-entries N        Limit to N entries (for quick testing)
#   --include-predictive    Include predictive.jsonl dataset
#   --counterfactual-test-size N  Limit counterfactual_test entries
#   --descriptive-size N   Limit descriptive entries
#   --predictive-size N    Limit predictive entries
#
# EXAMPLES:
#
#   # Basic evaluation with default settings
#   sbatch scripts/run_eval.sh \
#     --model "Qwen/Qwen3-VL-4B-Instruct" \
#     --dataset ds1
#
#   # Evaluation with custom parameters
#   sbatch scripts/run_eval.sh \
#     --model "Qwen/Qwen3-VL-4B-Thinking" \
#     --dataset ds1 \
#     --num-samples 1 \
#     --max-concurrent 256 \
#     --max-tokens 32768
#
#   # For causalpool-4B (merged checkpoint) with custom vLLM SIF path
#   sbatch scripts/run_eval.sh \
#     --model "causalpool-4B" \
#     --dataset ds1 \
#     --vllm-sif ~/scratch/vllm.sif
#
#   # Quick test with limited entries
#   sbatch scripts/run_eval.sh \
#     --model "Qwen/Qwen3-VL-4B-Instruct" \
#     --dataset ds1 \
#     --max-entries 10
#
# OUTPUT:
#   - SLURM output: outputs/slurm/auto_eval_<JOB_ID>.out
#   - SLURM error:  outputs/slurm/auto_eval_<JOB_ID>.err
#   - Server logs:  outputs/slurm/vllm_server_<MODEL>_<PORT>.log
#   - Evaluation results: See eval.py output for result locations
#
# MONITORING:
#   Check job status:     squeue -u $USER
#   View output:          tail -f outputs/slurm/auto_eval_<JOB_ID>.out
#   Cancel job:           scancel <JOB_ID>
#
# NOTES:
#   - The script automatically selects an idle GPU on the node
#   - Each job uses exactly 1 GPU
#   - Server cleanup happens automatically on completion or error
#   - For multiple models, use batch_eval.sh to submit multiple jobs at once
#
################################################################################

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
# Limit thread spawning for libraries (prevents excessive threads on login nodes)
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Activate virtual environment
source .venv/bin/activate

# Extract model name from arguments for job naming (if provided)
MODEL_NAME=""
for arg in "$@"; do
    if [[ "$arg" == "--model" ]]; then
        NEXT_IS_MODEL=true
    elif [[ -n "${NEXT_IS_MODEL:-}" ]]; then
        MODEL_NAME="$arg"
        break
    fi
done

# Update job name if model was provided (SLURM allows this before job starts)
if [[ -n "$MODEL_NAME" ]]; then
    # Sanitize model name for job name (replace / and - with _)
    SANITIZED_MODEL=$(echo "$MODEL_NAME" | sed 's/[\/-]/_/g')
    # Note: SLURM job name can't be changed after sbatch, but we can log it
    echo "Job name: auto_eval (model: $MODEL_NAME)"
fi

# Run automated evaluation
# All arguments passed to this script are forwarded to auto_eval.py
# The script will automatically:
# - Select an idle GPU
# - Launch vLLM server
# - Wait for server readiness
# - Run eval.py
# - Clean up and exit
#
# Additional arguments you can pass:
#   --max-entries N          : Limit to N entries (for quick testing)
#   --include-predictive     : Include predictive.jsonl dataset
#   --counterfactual-test-size N : Limit counterfactual_test entries
#   --descriptive-size N     : Limit descriptive entries
#   --predictive-size N      : Limit predictive entries
python scripts/auto_eval.py "$@"

# Job will terminate automatically after eval completes (or on error)

