#!/bin/bash
# Clone the upstream SC-GS implementation and apply the one build fix this
# project needs (gcc >= 13 no longer transitively includes <cstdint>).
# Usage: bash scripts/setup_upstream.sh
set -eo pipefail
cd "$(dirname "$0")/.."

if [ -d SC-GS ]; then
    echo "SC-GS/ already exists; nothing to do."
    exit 0
fi

git clone --recursive https://github.com/yihua7/SC-GS
git -C SC-GS checkout 3a9d2ad          # commit used for every result in this repo
git -C SC-GS submodule update --init --recursive
git -C SC-GS/submodules/diff-gaussian-rasterization apply \
    ../../../patches/diff-gaussian-rasterization-gcc13.patch
echo "Upstream ready. Next: see docs/SETUP.md for the conda environment."
