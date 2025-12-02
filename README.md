# Causal Pool

A video understanding benchmark for evaluating vision-language models on causal reasoning in pool table physics. The dataset contains multiple-choice questions about cue ball movements, requiring models to understand physical causality from video observations.

## Overview

Causal Pool evaluates models on four question types:
- **Descriptive**: What happened in the video?
- **Predictive**: What will happen next?
- **Counterfactual (velocity)**: What if the initial velocity changed?
- **Counterfactual (position)**: What if the initial position changed?

Each question includes a video of pool table physics simulation and multiple-choice options about ball trajectories, wall collisions, and pocket outcomes.

## Setup

```sh
uv sync
```

For SLURM cluster usage, see `documents/dev.md` for detailed setup instructions.

## Dataset Structure

Datasets are organized in `datasets/` with the following structure:

```
datasets/
└── <dataset_name>/
    ├── splits/
    │   ├── train-{question_type}.jsonl
    │   └── test-{question_type}.jsonl
    ├── shots/
    │   └── <shot_id>/
    │       └── video files (*-full.mp4, *-partial.mp4)
    └── raw_qa.jsonl
```

Each dataset entry contains:
- `video`: Shot identifier
- `question`: Question text with context
- `options`: List of multiple-choice options
- `ground_truth`: Correct answer(s) as indices
- `metadata.question_type`: One of the four question types

## Supervised Fine-Tuning (SFT)

Train models on the dataset using LoRA:

```sh
# Configure dataset and model in causal_pool/sft/train.py
python causal_pool/sft/train.py
```

Or use the SLURM script:
```sh
sbatch scripts/sft.sh
```

Training supports:
- Counterfactual training (position/velocity)
- Descriptive training
- LoRA with configurable rank and target modules
- Custom trainer with per-question accuracy metrics

## Evaluation

### Quick Start

Evaluate a model on a dataset:

```sh
# Start vLLM server (see documents/dev.md for full setup)
# Then run evaluation:
python causal_pool/eval/eval.py \
  --dataset <dataset_name> \
  --base-url "http://localhost:8000/v1" \
  --model "Qwen/Qwen3-VL-4B-Instruct" \
  --num-samples 1 \
  --max-concurrent 256 \
  --max-tokens 10
```

### Automated Evaluation (SLURM)

Use the automated evaluation script that handles vLLM server lifecycle:

```sh
# Via SLURM
sbatch scripts/run_eval.sh --model "Qwen/Qwen3-VL-4B-Instruct" --dataset <dataset_name>

# Directly on compute node
bash scripts/run_eval_direct.sh --model "Qwen/Qwen3-VL-4B-Instruct" --dataset <dataset_name>
```

### Batch Evaluation

Evaluate multiple models configured in `configs/eval/config.yaml`:

```sh
python scripts/batch_eval.py
```

### Manual Interactive Evaluation

For debugging individual questions:

```sh
python scripts/manual_eval.py -d <dataset_name> -m "Qwen/Qwen3-VL-4B-Instruct" -u http://localhost:8000/v1 --fps 15 -i 0
```

## Evaluation Configuration

The evaluation system uses Hydra-based configs in `configs/eval/`:

- `config.yaml`: Main config specifying which models to evaluate
- `models/`: Model-specific configurations
- `vllm/`: vLLM serving configurations
- `eval/`: Evaluation hyperparameters

See `configs/eval/README.md` for detailed configuration guide.

## Metrics

Two accuracy metrics are computed:
- **Per-question accuracy**: Exact match of predicted answer(s) with ground truth
- **Per-option accuracy**: Hamming distance between predicted and ground truth option sets

Results are saved as JSON files in `results/` with detailed per-question breakdowns.

## Project Structure

```
causal_pool/
├── causal_pool/          # Main package
│   ├── data/            # Dataset loading utilities
│   ├── eval/            # Evaluation scripts and utilities
│   ├── sft/             # Supervised fine-tuning code
│   ├── metrics.py       # Accuracy metrics
│   └── prompt_utils.py  # Prompt construction
├── configs/             # Hydra configuration files
├── datasets/            # Dataset files and splits
├── scripts/             # Utility scripts
│   ├── auto_eval.py    # Automated evaluation orchestrator
│   ├── batch_eval.py   # Batch evaluation submission
│   ├── manual_eval.py  # Interactive evaluation
│   └── process_dataset.py  # Dataset processing
├── outputs/            # Training outputs and logs
└── results/            # Evaluation results
```

## Key Scripts

- `scripts/auto_eval.py`: Orchestrates vLLM server launch and evaluation
- `scripts/batch_eval.py`: Submits multiple evaluation jobs
- `scripts/manual_eval.py`: Interactive evaluation for debugging
- `scripts/process_dataset.py`: Process raw QA data into train/test splits
- `scripts/merge_lora_ckpt.py`: Merge LoRA checkpoints with base model
- `scripts/plot_category_metrics.py`: Visualize evaluation results

## Supported Models

- Qwen3-VL (4B, 8B, 32B, 30B-A3B Instruct/Thinking variants)
- Custom fine-tuned CausalPool models

## Requirements

- Python >= 3.12
- CUDA-capable GPU
- vLLM for model serving (via Apptainer on SLURM clusters)
- See `pyproject.toml` for full dependency list

## Documentation

- `documents/dev.md`: Detailed development and cluster usage guide
- `configs/eval/README.md`: Evaluation configuration guide
- `tests/README.md`: Testing documentation
