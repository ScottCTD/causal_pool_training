# Causal Pool

## SFT

```sh
module load StdEnv/2023 gcc/12.3 cuda/12.6
uv sync
```

## Eval

```sh
module load StdEnv/2023 gcc/12.3 cuda/12.6
apptainer run --nv \
  --bind /scratch/scottc/:/scratch/scottc/ \
  --bind /home/scottc/links/:/home/scottc/links/ \
  --bind /scratch/scottc/cache/:/home/scottc/.cache \
  --env HF_HUB_OFFLINE=1 \
  vllm.sif \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 8192 \
```

```sh
uv run python eval/eval.py \
  --dataset 1k_simple \
  --base-url "http://trig0002:8000/v1" \
  --model "Qwen/Qwen3-VL-4B-Instruct" \
  --num-samples 16 \
  --max-concurrent 512 \
  --max-tokens 10 \
  --exclude-predictive \
```
