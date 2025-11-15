#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --output=sft_%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00

set -a
source .env
set +a

export WANDB_DISABLED=false
export WANDB_MODE=offline
export WANDB_ENTITY=csc2541-causality
export WANDB_PROJECT=csc2541-causality

export HF_HUB_OFFLINE=1

# export TOKENIZERS_PARALLELISM=false
# export RAYON_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=1

module load StdEnv/2023 gcc/12.3 cuda/12.6
source .venv/bin/activate

python causal_pool/sft/train.py
