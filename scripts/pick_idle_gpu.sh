#!/bin/bash
#
# Select the most idle GPU by checking utilization and memory usage.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=$(./scripts/pick_idle_gpu.sh)
#   # or
#   CUDA_VISIBLE_DEVICES=$(./scripts/pick_idle_gpu.sh) python my_script.py
#
# Output: GPU index (e.g., "0", "1", "2", "3")
# On error: defaults to "0"

set -euo pipefail

# Query GPU status via nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found, defaulting to GPU 0" >&2
    echo "0"
    exit 0
fi

# Get GPU information: index, utilization, memory used
gpu_info=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null || true)

if [ -z "$gpu_info" ]; then
    echo "WARNING: Failed to query nvidia-smi, defaulting to GPU 0" >&2
    echo "0"
    exit 0
fi

# Parse and sort GPUs by utilization (ascending), then memory used (ascending)
# Format: "index, utilization, memory_used"
best_gpu="0"
best_util=999
best_mem=999999

while IFS=', ' read -r gpu_idx utilization memory_used; do
    # Skip empty lines
    [ -z "$gpu_idx" ] && continue
    
    # Remove any whitespace
    gpu_idx=$(echo "$gpu_idx" | tr -d '[:space:]')
    utilization=$(echo "$utilization" | tr -d '[:space:]')
    memory_used=$(echo "$memory_used" | tr -d '[:space:]')
    
    # Skip if we don't have valid numbers
    if ! [[ "$utilization" =~ ^[0-9]+$ ]] || ! [[ "$memory_used" =~ ^[0-9]+$ ]]; then
        continue
    fi
    
    # Check if this GPU is better (lower utilization, or same utilization but lower memory)
    if [ "$utilization" -lt "$best_util" ] || \
       ([ "$utilization" -eq "$best_util" ] && [ "$memory_used" -lt "$best_mem" ]); then
        best_gpu="$gpu_idx"
        best_util="$utilization"
        best_mem="$memory_used"
    fi
done <<< "$gpu_info"

# Output the selected GPU index
echo "$best_gpu"

