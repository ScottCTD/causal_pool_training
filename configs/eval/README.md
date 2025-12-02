# Evaluation Configuration Guide

This directory contains Hydra-based configuration files for the evaluation pipeline.

## Structure

```
configs/eval/
├── config.yaml              # Main config (specifies which models to evaluate)
├── models/                  # Model-specific configs
│   ├── qwen_4b_instruct.yaml
│   ├── qwen_4b_thinking.yaml
│   ├── qwen_8b_instruct.yaml
│   ├── causalpool_4b.yaml
│   └── qwen_30b_a3b_instruct.yaml
├── vllm/                    # vLLM serving configs
│   ├── default.yaml         # Base vLLM config
│   ├── qwen_4b_instruct.yaml
│   ├── qwen_4b_thinking.yaml
│   ├── qwen_8b_instruct.yaml
│   ├── causalpool_4b.yaml
│   └── qwen_30b_a3b_instruct.yaml
└── eval/                    # Evaluation configs
    ├── default.yaml          # Base eval config
    ├── qwen_4b_instruct.yaml
    ├── qwen_4b_thinking.yaml
    ├── qwen_8b_instruct.yaml
    ├── causalpool_4b.yaml
    └── qwen_30b_a3b_instruct.yaml
```

## Usage

### 1. Specifying Models to Evaluate

Edit `config.yaml` to specify which models to evaluate:

```yaml
models_to_evaluate:
  - qwen_4b_instruct
  - qwen_8b_instruct
  # - qwen_4b_thinking  # Commented out = not evaluated
```

### 2. Adding a New Model

To add a new model, create three config files:

#### Step 1: Create model config (`models/your_model.yaml`)

```yaml
model_name: "Your/Model-Name"

# vLLM serving config (references configs/eval/vllm/)
vllm_config: your_model

# Evaluation config (references configs/eval/eval/)
eval_config: default

# Model-specific eval overrides (merged with eval config)
eval_overrides:
  dataset: "ds1"
  # num_samples: 1
  # max_concurrent: 256
```

#### Step 2: Create vLLM config (`vllm/your_model.yaml`)

```yaml
defaults:
  - default

model: "Your/Model-Name"
host: "0.0.0.0"
tensor_parallel_size: 1
gpu_memory_utilization: 0.9
max_model_len: 8192
max_num_seqs: 512
enforce_eager: true
```

#### Step 3: Create eval config (`eval/your_model.yaml`)

```yaml
defaults:
  - default

hyperparameters:
  temperature: 0.8
  top_k: 20
  top_p: 0.8
  repetition_penalty: 1.0
  presence_penalty: 1.5
```

#### Step 4: Add to main config

Add your model to `config.yaml`:

```yaml
models_to_evaluate:
  - your_model
```

### 3. Modifying vLLM Serving Arguments

Edit the corresponding file in `vllm/`:

```yaml
# vllm/qwen_4b_instruct.yaml
defaults:
  - default

model: "Qwen/Qwen3-VL-4B-Instruct"
gpu_memory_utilization: 0.95  # Changed from default
max_model_len: 16384          # Changed from default
```

### 4. Modifying Evaluation Arguments

Edit the corresponding file in `eval/` or use `eval_overrides` in the model config:

```yaml
# Option 1: Edit eval/qwen_4b_instruct.yaml
defaults:
  - default

hyperparameters:
  temperature: 0.9  # Changed from default

# Option 2: Use eval_overrides in models/qwen_4b_instruct.yaml
eval_overrides:
  dataset: "ds1"
  num_samples: 2        # Override default
  max_concurrent: 128   # Override default
```

## Running Evaluations

### Single Model Evaluation

#### Via SLURM (recommended for batch jobs)
```bash
sbatch scripts/run_eval.sh --model "Qwen/Qwen3-VL-4B-Instruct" --dataset ds2
```

#### Directly on Compute Node (when already on a compute node)
```bash
# Option 1: Use the wrapper script
bash scripts/run_eval_direct.sh --model "Qwen/Qwen3-VL-4B-Instruct" --dataset ds2

# Option 2: Set up environment manually
source scripts/setup_eval_env.sh
python scripts/auto_eval.py --model "Qwen/Qwen3-VL-4B-Instruct" --dataset ds2
```

**Note**: When running directly on a compute node:
- Make sure you're on a compute node with GPU access (e.g., via `srun` or interactive session)
- The script will use all available GPUs (or set `CUDA_VISIBLE_DEVICES` manually)
- Port conflicts: Make sure the port (default 8000) is not already in use

### Batch Evaluation (Multiple Models)

```bash
python scripts/batch_eval.py
```

This will read `configs/eval/config.yaml` and submit jobs for all models listed in `models_to_evaluate`.

## Configuration Precedence

1. **vLLM configs**: Model-specific configs override `default.yaml`
2. **Eval configs**: Model-specific configs override `default.yaml`
3. **Model eval_overrides**: Override both eval configs (highest priority)

## Common Settings

### vLLM Serving Arguments

- `model`: Model name (HF path or local path)
- `host`: Server host (usually "0.0.0.0")
- `port`: Server port (auto-assigned in batch mode)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `gpu_memory_utilization`: GPU memory utilization (0.0-1.0)
- `max_model_len`: Maximum sequence length
- `max_num_seqs`: Maximum number of sequences in batch
- `enforce_eager`: Use eager mode (disable CUDA graphs)
- `trust_remote_code`: Trust remote code (for some models)
- `reasoning_parser`: Reasoning parser (e.g., "qwen3" for thinking models)
- `enable_expert_parallel`: Enable expert parallelism (for MoE models)

### Evaluation Arguments

- `dataset`: Dataset name (e.g., "ds1")
- `num_samples`: Number of samples per question
- `max_concurrent`: Maximum concurrent requests
- `max_tokens`: Maximum tokens for generation (null = model default)
- `temperature`: Sampling temperature (null = model default)
- `max_entries`: Limit number of entries (null = all)
- `counterfactual_velocity_size`: Limit counterfactual_velocity entries
- `counterfactual_position_size`: Limit counterfactual_position entries
- `descriptive_size`: Limit descriptive entries
- `predictive_size`: Limit predictive entries
- `include_predictive`: Include predictive.jsonl dataset
- `fps`: Frames per second for video processing

### Hyperparameters

- `temperature`: Sampling temperature
- `top_k`: Top-k sampling
- `top_p`: Top-p (nucleus) sampling
- `repetition_penalty`: Repetition penalty
- `presence_penalty`: Presence penalty

