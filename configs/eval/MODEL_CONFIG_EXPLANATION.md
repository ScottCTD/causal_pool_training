# Model Configuration Files Explained

Each file in `configs/eval/models/` defines how a specific model should be evaluated. Here's what each part does:

## Structure

```yaml
# models/qwen_4b_instruct.yaml
model_name: "Qwen/Qwen3-VL-4B-Instruct"  # The actual model name/identifier

vllm_config: qwen_4b_instruct  # Which vLLM serving config to use (from vllm/)
eval_config: default            # Which eval config to use (from eval/)

eval_overrides:                 # Model-specific parameter overrides
  max_tokens: 32768            # Override max tokens for this model
  temperature: 0.9             # Override temperature for this model
```

## What Each Field Does

### 1. `model_name`
- **Purpose**: The actual model identifier used by the evaluation scripts
- **Example**: `"Qwen/Qwen3-VL-4B-Instruct"`, `"causalpool-4B"`
- **Used by**: `auto_eval.py` to identify which model to serve

### 2. `vllm_config`
- **Purpose**: Points to a vLLM serving configuration file
- **Location**: `configs/eval/vllm/{vllm_config}.yaml`
- **Contains**: vLLM server arguments like:
  - `gpu_memory_utilization`
  - `max_model_len`
  - `max_num_seqs`
  - `tensor_parallel_size`
  - etc.
- **Why separate**: Different models need different serving parameters (memory, sequence length, etc.)

### 3. `eval_config`
- **Purpose**: Points to an evaluation configuration file
- **Location**: `configs/eval/eval/{eval_config}.yaml`
- **Contains**: Model hyperparameters for evaluation:
  - `temperature`
  - `top_k`
  - `top_p`
  - `repetition_penalty`
  - `presence_penalty`
- **Why separate**: Different models may need different sampling parameters

### 4. `eval_overrides`
- **Purpose**: Model-specific overrides for evaluation parameters
- **Precedence**: Highest priority (overrides both `eval_config` and global `config.yaml`)
- **Common uses**:
  - `max_tokens`: Override max generation tokens (e.g., thinking models need more)
  - `temperature`: Override sampling temperature
  - `num_samples`: Override number of samples per question
  - `max_concurrent`: Override concurrent request limit

## Config Precedence (Highest to Lowest)

1. **`eval_overrides`** in `models/{model}.yaml` (model-specific overrides)
2. **`dataset`** in `config.yaml` (global dataset setting)
3. **`eval/{eval_config}.yaml`** (model's eval config, e.g., `eval/qwen_4b_instruct.yaml`)
4. **`eval/default.yaml`** (base eval config)

## Example: Qwen3-VL-4B-Thinking

```yaml
# models/qwen_4b_thinking.yaml
model_name: "Qwen/Qwen3-VL-4B-Thinking"
vllm_config: qwen_4b_thinking      # Uses vllm/qwen_4b_thinking.yaml
eval_config: default                # Uses eval/default.yaml
eval_overrides:
  max_tokens: 32768                 # Override: thinking models need more tokens
```

This model:
- Uses `vllm/qwen_4b_thinking.yaml` for serving (which has `max_model_len: 40960` and `reasoning_parser: "qwen3"`)
- Uses `eval/default.yaml` for hyperparameters
- Overrides `max_tokens` to 32768 (higher than default) for longer reasoning outputs

## Adding Model-Specific Sampling Parameters

If you want different sampling parameters for a specific model, you have two options:

### Option 1: Use `eval_overrides` (simple, for one-off changes)
```yaml
# models/qwen_4b_instruct.yaml
eval_overrides:
  max_tokens: 16384
  temperature: 0.9
```

### Option 2: Create a model-specific eval config (better for complex differences)
1. Create `eval/qwen_4b_instruct.yaml`:
```yaml
defaults:
  - default

hyperparameters:
  temperature: 0.9
  top_p: 0.9
```

2. Update `models/qwen_4b_instruct.yaml`:
```yaml
eval_config: qwen_4b_instruct  # Use the new config
```

## Dataset Setting

The `dataset` parameter is now set globally in `config.yaml`:
```yaml
# config.yaml
dataset: "ds1"  # Applied to all models
```

You can still override it per-model in `eval_overrides` if needed, but typically you want the same dataset for all models in a batch run.

