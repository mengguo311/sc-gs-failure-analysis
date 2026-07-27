#!/bin/bash
# Shared environment for all experiment scripts. Source this first.
export PATH=/home/u00134/miniconda3/envs/scgs/bin:$PATH
export CUDA_HOME=/home/u00134/miniconda3/envs/scgs
export TORCH_CUDA_ARCH_LIST="7.5"
PROJ=/home/u00134/media-paper
SCGS=$PROJ/SC-GS
DATA=$PROJ/data
OUT=$PROJ/outputs
RESULTS=$PROJ/results
mkdir -p "$OUT" "$RESULTS"
