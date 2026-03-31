#!/bin/bash
#SBATCH --job-name=extract-token
#SBATCH --output=sbatch/%j_%x.out
#SBATCH --error=sbatch/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --export=ALL

# Extract VQ tokens from an ImageNet split at a given resolution.
#
# Usage:
#   bash scripts/tokens/extract_token.sh        # Local (uses all visible GPUs)
#   sbatch scripts/tokens/extract_token.sh      # Slurm
set -e

source $HOME/.bashrc

# -------------------------------
#  Config
# -------------------------------
SPLIT_NAME="val"
RESOLUTION=1024

# -------------------------------
#  Environment setup
# -------------------------------
if [ -n "$SLURM_JOB_ID" ] && command -v module &> /dev/null; then
    module load cuda/12.6 cudatoolkit/24.11_12.6 gcc-native/13.2
    module load brics/nccl brics/aws-ofi-nccl 2>/dev/null || true
fi

# -------------------------------
#  Distributed setup
# -------------------------------
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    export GPUS_PER_NODE=1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# -------------------------------
#  Install
# -------------------------------
uv sync

# Create sbatch output dir if needed
if [ -n "$SLURM_JOB_ID" ]; then
    mkdir -p sbatch
fi

# -------------------------------
#  Generate image metadata
# -------------------------------
uv run scripts/tokens/generate_meta.py \
    --mode image_meta \
    --split_dir data/imagenet-1k/ImageNet/${SPLIT_NAME} \
    --output data/imagenet_${SPLIT_NAME}_meta.json

# -------------------------------
#  Extract tokens
# -------------------------------
uv run torchrun \
    --standalone \
    --nproc_per_node=${GPUS_PER_NODE} \
    simpar/data/extract_token.py \
        --dataset_type "image" \
        --dataset_name "imagenet_${SPLIT_NAME}" \
        --code_path data/imagenet_tokens \
        --gen_data_path data/imagenet_${SPLIT_NAME}_meta.json \
        --gen_image_folder "" \
        --gen_resolution ${RESOLUTION} \
        --vq_model_ckpt "./checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16"

# -------------------------------
#  Generate token metadata
# -------------------------------
uv run scripts/tokens/generate_meta.py \
    --mode token_meta \
    --tokens_dir data/imagenet_tokens/imagenet_${SPLIT_NAME} \
    --resolution ${RESOLUTION} \
    --output data/imagenet_${SPLIT_NAME}_${RESOLUTION}_token_meta.json
