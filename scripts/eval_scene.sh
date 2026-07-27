#!/bin/bash
# Evaluate a trained scene: render test views + compute PSNR/SSIM/LPIPS.
# Usage: scripts/eval_scene.sh <scene> [node_num] [tag]
set -eo pipefail
source "$(dirname "$0")/env.sh"

SCENE=${1:?usage: eval_scene.sh <scene> [node_num] [tag]}
NODE_NUM=${2:-512}
TAG=${3:-n$NODE_NUM}
MODEL_DIR=$OUT/${SCENE}_${TAG}_node
LOG=$OUT/${SCENE}_${TAG}_eval.log

echo "[eval] scene=$SCENE model=$MODEL_DIR" | tee "$LOG"
CUDA_VISIBLE_DEVICES=0 python "$SCGS/render.py" \
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
