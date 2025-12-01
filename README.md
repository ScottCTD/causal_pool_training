# Causal Pool

## SFT

```sh
module load StdEnv/2023 gcc/12.3 cuda/12.6
uv sync
```

## Eval

```sh
salloc --nodes=1 --gpus-per-node=1 --time 0-08:00:00 &

module load StdEnv/2023 gcc/12.3 cuda/12.6
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=$(./scripts/pick_idle_gpu.sh)

apptainer run --nv \
  --bind /scratch/scottc/:/scratch/scottc/ \
  --bind /home/scottc/links/:/home/scottc/links/ \
  --bind /scratch/scottc/cache/:/home/scottc/.cache \
  --bind /scratch/scottc/cache/triton/:/home/scottc/.triton/ \
  --env HF_HUB_OFFLINE=1 \
  ~/scratch/vllm.sif \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --enforce-eager \
  --media-io-kwargs.video.num_frames -1 \

apptainer run --nv \
  --bind /scratch/scottc/:/scratch/scottc/ \
  --bind /home/scottc/links/:/home/scottc/links/ \
  --bind /scratch/scottc/cache/:/home/scottc/.cache \
  --bind /scratch/scottc/cache/triton/:/home/scottc/.triton/ \
  --env HF_HUB_OFFLINE=1 \
  vllm.sif \
  --model Qwen/Qwen3-VL-4B-Thinking \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 40960 \
  --max-num-seqs 256 \
  --reasoning-parser qwen3 \
  --enforce-eager \
  --media-io-kwargs.video.num_frames -1 \

apptainer run --nv \
  --bind /scratch/scottc/:/scratch/scottc/ \
  --bind /home/scottc/links/:/home/scottc/links/ \
  --bind /scratch/scottc/cache/:/home/scottc/.cache \
  --bind /scratch/scottc/cache/triton/:/home/scottc/.triton/ \
  --env HF_HUB_OFFLINE=1 \
  vllm.sif \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 32768 \
  --max-num-seqs 512 \
  --enforce-eager \
  --media-io-kwargs.video.num_frames -1 \

apptainer run --nv \
  --bind /scratch/scottc/:/scratch/scottc/ \
  --bind /home/scottc/links/:/home/scottc/links/ \
  --bind /scratch/scottc/cache/:/home/scottc/.cache \
  --bind /scratch/scottc/cache/triton/:/home/scottc/.triton/ \
  --env HF_HUB_OFFLINE=1 \
  vllm.sif \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --enforce-eager \
  --media-io-kwargs.video.num_frames -1 \

apptainer run --nv \
  --bind /scratch/scottc/:/scratch/scottc/ \
  --bind /home/scottc/links/:/home/scottc/links/ \
  --bind /scratch/scottc/cache/:/home/scottc/.cache \
  --bind /scratch/scottc/cache/triton/:/home/scottc/.triton/ \
  --env HF_HUB_OFFLINE=1 \
  ~/scratch/vllm.sif \
  --model OpenGVLab/InternVL3_5-4B \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 40960 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --enforce-eager \

apptainer run --nv \
  --bind /scratch/scottc/:/scratch/scottc/ \
  --bind /home/scottc/links/:/home/scottc/links/ \
  --bind /scratch/scottc/cache/:/home/scottc/.cache \
  --bind /scratch/scottc/cache/triton/:/home/scottc/.triton/ \
  --env HF_HUB_OFFLINE=1 \
  ~/scratch/vllm.sif \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --enforce-eager \
  --enable-expert-parallel \
  --media-io-kwargs.video.num_frames -1 \
```

```sh
set -a
source .env
set +a

source .venv/bin/activate

uv run python causal_pool/eval/eval.py \
  --dataset ds2 \
  --base-url "http://trig0001:8000/v1" \
  --model "Qwen/Qwen3-VL-4B-Instruct" \
  --num-samples 1 \
  --max-concurrent 256 \
  --max-tokens 10 \
  --counterfactual-velocity-size 0 \
  --counterfactual-position-size 0 \
  --descriptive-size 256 \
  --predictive-size 256 \
  --fps 8 \

uv run python causal_pool/eval/eval.py \
  --dataset 1k_simple \
  --base-url "http://trig0006:8000/v1" \
  --model "Qwen/Qwen3-VL-4B-Thinking" \
  --num-samples 1 \
  --max-concurrent 64 \
  --max-tokens 32768 \

uv run python causal_pool/eval/eval.py \
  --dataset ds2 \
  --base-url "http://trig0005:8000/v1" \
  --model "Qwen/Qwen3-VL-8B-Instruct" \
  --num-samples 1 \
  --max-concurrent 128 \
  --max-tokens 10 \
  --counterfactual-velocity-size 0 \
  --counterfactual-position-size 0 \
  --descriptive-size 256 \
  --predictive-size 256 \

uv run python causal_pool/eval/eval.py \
  --dataset ds2 \
  --base-url "http://trig0005:8000/v1" \
  --model "Qwen/Qwen3-VL-32B-Instruct" \
  --num-samples 1 \
  --max-concurrent 128 \
  --max-tokens 10 \
  --counterfactual-velocity-size 0 \
  --counterfactual-position-size 0 \
  --descriptive-size 256 \
  --predictive-size 256 \

uv run python eval/eval.py \
  --dataset 1k_simple \
  --base-url "http://trig0006:8000/v1" \
  --model "OpenGVLab/InternVL3_5-4B" \
  --num-samples 1 \
  --max-concurrent 512 \
  --max-tokens 10 \

uv run python causal_pool/eval/eval.py \
  --dataset ds1 \
  --base-url "http://trig0001:8000/v1" \
  --model "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --num-samples 1 \
  --max-concurrent 256 \
  --max-tokens 10 \
  -e 1 \
```

Manual eval:
```sh
python scripts/manual_eval.py -d ds2 -m "Qwen/Qwen3-VL-32B-Instruct" -u http://trig0005:8000/v1 --fps 15 -i 0
```

```sh
HF_HUB_OFFLINE=1 python scripts/merge_lora_ckpt.py \
  -b "Qwen/Qwen3-VL-4B-Instruct" \
  -c "outputs_sft/checkpoint-810/" \

apptainer run --nv \
  --bind /scratch/scottc/:/scratch/scottc/ \
  --bind /home/scottc/links/:/home/scottc/links/ \
  --bind /scratch/scottc/cache/:/home/scottc/.cache \
  --bind /scratch/scottc/cache/triton/:/home/scottc/.triton/ \
  --env HF_HUB_OFFLINE=1 \
  ~/scratch/vllm.sif \
  --model outputs_sft/checkpoint-810/merged/ \
  --served-model-name causalpool-4B \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 8192 \
  --enforce-eager \

python causal_pool/eval/eval.py \
  --dataset 1k_simple \
  --base-url "http://trig0032:8000/v1" \
  --model "causalpool-4B" \
  --num-samples 1 \
  --max-concurrent 256 \
  --max-tokens 10 \
```

```sh
export OPENBLAS_NUM_THREADS=1
```

Start Jupyter server:
```sh
source .venv/bin/activate
module load StdEnv/2023 gcc/12.3 cuda/12.6
export HF_HUB_OFFLINE=1
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```