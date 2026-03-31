#!/bin/bash
# Install the repo and the correct PyTorch CUDA variant for the current environment.
# Source this file from other scripts — do not execute it directly.
#
# Local dev:  CUDA 13.0 → torch+cu130
# Slurm:      CUDA 12.6 → torch+cu126

set -e

uv sync --extra train

if [ -n "$SLURM_JOB_ID" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
    TORCH_CUDA="cu126"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
    TORCH_CUDA="cu130"
fi

uv pip install \
    torch==2.11.0+${TORCH_CUDA} \
    torchvision==0.26.0+${TORCH_CUDA} \
    --index-url "${TORCH_INDEX}"
