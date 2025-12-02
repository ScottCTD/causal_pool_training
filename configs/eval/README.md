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

To completely add a new model to the evaluation system, you need to:

1. Create config files (vLLM and eval configs)
2. Add model name mappings in Python code (required for `auto_eval.py`)
3. Optionally create a model config file (for batch evaluation)

**Important**: The model name you use must match exactly across all files (including the Python mappings).

#### Step 1: Create vLLM config (`vllm/your_model.yaml`)

Create a vLLM serving configuration file. Choose a config name based on your model (e.g., `qwen_32b_instruct.yaml` for `Qwen/Qwen3-VL-32B-Instruct`):

```yaml
# vllm/qwen_32b_instruct.yaml
defaults:
  - default

model: "Qwen/Qwen3-VL-32B-Instruct"
host: "0.0.0.0"
tensor_parallel_size: 1
gpu_memory_utilization: 0.9
max_model_len: 8192
max_num_seqs: 512
enforce_eager: true
```

**Note**: The `model` field should contain the exact model identifier (HuggingFace path or local path).

#### Step 2: Create eval config (`eval/your_model.yaml`)

Create an evaluation configuration file with the same name:

```yaml
# eval/qwen_32b_instruct.yaml
defaults:
  - default

hyperparameters:
  temperature: 0.8
  top_k: 20
  top_p: 0.8
  repetition_penalty: 1.0
  presence_penalty: 1.5
```

#### Step 3: Add model name mappings in Python code

**CRITICAL**: You must add the model name mapping in two Python files, otherwise you'll get an "Unknown model" error.

##### 3a. Add to `scripts/auto_eval.py`

Edit the `load_vllm_config` function and add your model to the `model_to_config` dictionary:

```python
# scripts/auto_eval.py (around line 49-58)
model_to_config = {
    "Qwen/Qwen3-VL-4B-Instruct": "qwen_4b_instruct",
    "Qwen/Qwen3-VL-4B-Thinking": "qwen_4b_thinking",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen_8b_instruct",
    "CausalPool-4B-cf": "causalpool_4b_cf",
    "CausalPool-4B-desc": "causalpool_4b_desc",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen_30b_a3b_instruct",
    "Qwen/Qwen3-VL-32B-Instruct": "qwen_32b_instruct",  # <-- Add your model here
    "OpenGVLab/InternVL3_5-4B": "internvl3_5_4b",
}
```

Also update the docstring to document the new mapping:

```python
"""
Maps model names to config names:
- "Qwen/Qwen3-VL-4B-Instruct" -> "qwen_4b_instruct"
...
- "Qwen/Qwen3-VL-32B-Instruct" -> "qwen_32b_instruct"  # <-- Add here too
...
"""
```

##### 3b. Add to `causal_pool/eval/eval_utils.py`

Edit the `MODEL_TO_EVAL_CONFIG` dictionary:

```python
# causal_pool/eval/eval_utils.py (around line 25-34)
MODEL_TO_EVAL_CONFIG: Dict[str, str] = {
    "Qwen/Qwen3-VL-4B-Instruct": "qwen_4b_instruct",
    "Qwen/Qwen3-VL-4B-Thinking": "qwen_4b_thinking",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen_8b_instruct",
    "CausalPool-4B-cf": "causalpool_4b_cf",
    "CausalPool-4B-desc": "causalpool_4b_desc",
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "qwen_30b_a3b_instruct",
    "Qwen/Qwen3-VL-32B-Instruct": "qwen_32b_instruct",  # <-- Add your model here
    "OpenGVLab/InternVL3_5-4B": "default",
}
```

**Important**: 
- The model name (left side) must match exactly what you'll use in the command line (e.g., `--model "Qwen/Qwen3-VL-32B-Instruct"`)
- The config name (right side) must match the YAML filename without `.yaml` (e.g., `qwen_32b_instruct`)

#### Step 4: (Optional) Create model config for batch evaluation (`models/your_model.yaml`)

If you plan to use batch evaluation (`scripts/batch_eval.py`), create a model config file:

```yaml
# models/qwen_32b_instruct.yaml
model_name: "Qwen/Qwen3-VL-32B-Instruct"

# vLLM serving config (references configs/eval/vllm/)
vllm_config: qwen_32b_instruct

# Evaluation config (references configs/eval/eval/)
eval_config: qwen_32b_instruct

# Model-specific eval overrides (merged with eval config)
eval_overrides:
  dataset: "ds2"
  # num_samples: 1
  # max_concurrent: 256
```

#### Step 5: (Optional) Add to main config for batch evaluation

If using batch evaluation, add your model to `config.yaml`:

```yaml
models_to_evaluate:
  - qwen_32b_instruct  # Use the config name, not the model name
```

#### Summary Checklist

When adding a new model, ensure you've completed:

- [ ] Created `configs/eval/vllm/your_model.yaml` with vLLM serving settings
- [ ] Created `configs/eval/eval/your_model.yaml` with evaluation hyperparameters
- [ ] Added model mapping in `scripts/auto_eval.py` → `model_to_config` dictionary
- [ ] Added model mapping in `causal_pool/eval/eval_utils.py` → `MODEL_TO_EVAL_CONFIG` dictionary
- [ ] (Optional) Created `configs/eval/models/your_model.yaml` for batch evaluation
- [ ] (Optional) Added model to `configs/eval/config.yaml` → `models_to_evaluate` list

**Common mistake**: Forgetting to add the Python mappings (Step 3) will result in an "Unknown model" error even if the config files exist.

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

