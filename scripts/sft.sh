export WANDB_DISABLED=false
# export WANDB_MODE=offline
export WANDB_ENTITY=csc2541-causality
export WANDB_PROJECT=csc2541-causality

export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=1

module load StdEnv/2023 gcc/12.3 cuda/12.6
source .venv/bin/activate

uv run python sft/train.py
