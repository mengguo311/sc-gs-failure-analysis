#!/bin/bash
# Sequential training queue after the first jumpingjacks run:
#   Phase 1: hook, mutant (node_num=512)
#   Phase 2B: jumpingjacks with node_num 64, 128, 1024, 2048
# Each run is evaluated right after training so metrics accumulate incrementally.
set -eo pipefail
cd "$(dirname "$0")/.."

for SCENE in hook mutant; do
    bash scripts/train_scene.sh $SCENE 512
    bash scripts/eval_scene.sh  $SCENE 512
done

for N in 64 128 1024 2048; do
    bash scripts/train_scene.sh jumpingjacks $N
    bash scripts/eval_scene.sh  jumpingjacks $N
done

echo "[queue] all remaining trainings complete: $(date -Is)"
