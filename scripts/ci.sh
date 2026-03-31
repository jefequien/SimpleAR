#!/bin/bash
#SBATCH --job-name=ci
#SBATCH --output=sbatch/%j_%x.out
#SBATCH --error=sbatch/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --export=ALL

# CI: install the repo and run tests.
#
# Usage:
#   bash scripts/ci.sh       # Local
#   sbatch scripts/ci.sh     # Slurm
set -e

source $HOME/.bashrc

# -------------------------------
#  Environment setup
# -------------------------------
if command -v module &> /dev/null; then
    module load cuda/12.6 cudatoolkit/24.11_12.6 gcc-native/13.2
fi

# Create sbatch output dir if needed
if [ -n "$SLURM_JOB_ID" ]; then
    mkdir -p sbatch
fi

# -------------------------------
#  Environment info
# -------------------------------
echo "=== Environment ==="
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "User:        $(whoami)"
echo "Python:      $(uv run python --version 2>&1)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-all}"
uv run python -c "import torch; print('PyTorch:    ', torch.__version__); print('CUDA avail: ', torch.cuda.is_available()); [print(f'  GPU {i}:    {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "PyTorch:     not installed yet"
echo "Driver:      $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
echo "==================="

# -------------------------------
#  Install
# -------------------------------
if [ -n "$SLURM_JOB_ID" ]; then
    uv sync --extra train --extra cu126
else
    uv sync --extra train --extra cu130
fi

# -------------------------------
#  Test
# -------------------------------
uv run pytest tests/ -v
