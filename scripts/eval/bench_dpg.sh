CKPT_PATH=Daniel0724/SimpleAR-0.5B-RL # DPG-Bench score: 79.6609256969302
SAVE_FOLDER=simpar_0.5B_rl
CFG_SCALE=6.0

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} uv run python -m simpar.eval.model_t2i \
    --model-path ${CKPT_PATH} \
    --save_dir "./visualize/${SAVE_FOLDER}" \
    --ann_path "./eval/ELLA/dpg_bench/prompts" \
    --vq-model "cosmos" \
    --vq-model-ckpt "./checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16" \
    --image-size 1024 \
    --batch-size 1 \
    --top_k 64000 \
    --top_p 1.0 \
    --temperature 1.0 \
    --benchmark "dpg" \
    --num-images-per-prompt 4 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --cfg-scale $CFG_SCALE &
done
wait