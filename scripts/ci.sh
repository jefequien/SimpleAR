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
#  Install
# -------------------------------
source scripts/install_env.sh

# -------------------------------
#  Test
# -------------------------------
uv run pytest tests/ -v
