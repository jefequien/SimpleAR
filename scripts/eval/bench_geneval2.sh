CKPT_PATH=Daniel0724/SimpleAR-1.5B-RL # Overall score (avg. over tasks): 0.62568
SAVE_FOLDER=simpar_1.5B_rl
CFG_SCALE=6.0

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} uv run python -m simpar.eval.model_t2i \
    --model-path ${CKPT_PATH} \
    --save_dir ./outputs/visualize2/${CKPT_PATH} \
    --ann_path "./eval/geneval/prompts/evaluation_metadata.jsonl" \
    --vq-model "cosmos" \
    --vq-model-ckpt "./checkpoints/Cosmos-1.0-Tokenizer-DV8x16x16" \
    --image-size 1024 \
    --batch-size 1 \
    --temperature 1.0 \
    --top_k 64000 \
    --top_p 1.0 \
    --benchmark "geneval" \
    --num-images-per-prompt 4 \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --vllm_serving \
    --cfg-scale $CFG_SCALE &
done
wait