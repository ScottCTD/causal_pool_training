#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --output=outputs/slurm/test_cuda_%j.out
#SBATCH --error=outputs/slurm/test_cuda_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:05:00
# Quick test to check if SLURM sets CUDA_VISIBLE_DEVICES

echo "=========================================="
echo "Testing CUDA_VISIBLE_DEVICES from SLURM"
echo "=========================================="
echo ""
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $SLURM_NODELIST"
echo "SLURM GPUs requested: --gpus-per-node=1"
echo ""
echo "Environment variables:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<NOT SET>}"
echo ""
echo "SLURM GPU-related environment variables:"
env | grep -i "SLURM.*GPU" || echo "  (none found)"
echo ""
echo "All CUDA-related environment variables:"
env | grep -i cuda || echo "  (none found)"
echo ""
echo "GPU information from nvidia-smi:"
if command -v nvidia-smi &> /dev/null; then
    echo "  All visible GPUs (formatted):"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits
    echo ""
    echo "  Raw nvidia-smi output:"
    nvidia-smi
    echo ""
    echo "  Number of GPUs visible to this process:"
    python3 -c "import os; print(f\"    CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}\")" 2>/dev/null || echo "    (Python not available)"
else
    echo "  nvidia-smi not available"
fi
echo ""
echo "=========================================="
echo "Test complete"
echo "=========================================="

