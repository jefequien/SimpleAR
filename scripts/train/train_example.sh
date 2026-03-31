#!/bin/bash
#SBATCH --job-name=sft-imagenet
#SBATCH --output=sbatch/%j_%x.out
#SBATCH --error=sbatch/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --export=ALL

# SFT training on ImageNet val tokens.
#
# Usage:
#   bash scripts/train/train_example.sh        # Local (uses all visible GPUs)
#   sbatch scripts/train/train_example.sh      # Slurm
set -e

source $HOME/.bashrc

# -------------------------------
#  Config
# -------------------------------
PROMPT_VERSION="qwen_1_5"
LLM_VERSION="Daniel0724/SimpleAR-0.5B-SFT"
DATE=$(date +%Y-%m-%d-%H-%M-%S)
RUN_NAME="${DATE}_sft_imagenet_val"

# -------------------------------
#  Environment setup
# -------------------------------
if command -v module &> /dev/null; then
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------------------
#  Install
# -------------------------------
if [ -n "$SLURM_JOB_ID" ]; then
    uv sync --extra train --extra cu126
else
    uv sync --extra train --extra cu130
fi

# Persist compiled Triton kernels in the repo's .cache dir (portable across machines)
# SLURM_SUBMIT_DIR is set by sbatch to the submission directory; fall back to $PWD for local runs.
export TORCHINDUCTOR_CACHE_DIR="${SLURM_SUBMIT_DIR:-$PWD}/.cache/torchinductor"

# Create sbatch output dir if needed
if [ -n "$SLURM_JOB_ID" ]; then
    mkdir -p sbatch
fi

# -------------------------------
#  Launch training
# -------------------------------
uv run torchrun \
    --standalone \
    --nproc_per_node=${GPUS_PER_NODE} \
    simpar/train/train_mem.py \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --gen_data_path data/imagenet_val_1024_token_meta.json \
    --gen_image_folder "" \
    --sample_short True \
    --mm_tunable_parts="mm_language_model" \
    --p_drop_cond 0.1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${RUN_NAME} \
    --output_dir outputs/training/${RUN_NAME} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4352 \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --report_to wandb \
    --attn_implementation sdpa \
    --log_images_every 100
