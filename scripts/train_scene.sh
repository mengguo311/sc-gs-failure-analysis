#!/bin/bash
# Train one D-NeRF scene with the paper's standard config.
# Usage: scripts/train_scene.sh <scene> [node_num] [tag]
#   scene:    jumpingjacks | hook | mutant | ...
#   node_num: default 512 (paper config)
#   tag:      output subdir suffix; default "n<node_num>"
set -eo pipefail
source "$(dirname "$0")/env.sh"

SCENE=${1:?usage: train_scene.sh <scene> [node_num] [tag]}
NODE_NUM=${2:-512}
TAG=${3:-n$NODE_NUM}
MODEL_DIR=$OUT/${SCENE}_${TAG}          # train_gui.py appends "_node" automatically
LOG=$OUT/${SCENE}_${TAG}_train.log
mkdir -p "$OUT"

echo "[train] scene=$SCENE node_num=$NODE_NUM -> ${MODEL_DIR}_node"
echo "[train] start: $(date -Is)" | tee "$LOG"

# Peak GPU memory sampler (30 s period)
GPULOG=$OUT/${SCENE}_${TAG}_gpumem.csv
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$GPULOG"; sleep 30; done ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null || true" EXIT

START=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python "$SCGS/train_gui.py" \
    --source_path "$DATA/$SCENE" \
    --model_path "$MODEL_DIR" \
    --deform_type node \
    --node_num "$NODE_NUM" \
    --hyper_dim 8 \
    --is_blender \
    --eval \
    --gt_alpha_mask_as_scene_mask \
    --local_frame \
    --resolution 2 \
    --W 800 --H 800 \
    2>&1 | tee -a "$LOG"
END=$(date +%s)

echo "[train] end: $(date -Is)" | tee -a "$LOG"
echo "[train] wall_clock_seconds: $((END-START))" | tee -a "$LOG"
echo "[train] peak_gpu_mem_MiB: $(sort -n "$GPULOG" | tail -1)" | tee -a "$LOG"
