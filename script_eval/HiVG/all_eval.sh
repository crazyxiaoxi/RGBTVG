#!/bin/bash

# 批量评估脚本：遍历 3 个数据集 × 3 个模态（HiVG ViT-B）

DATASETS=${DATASETS:-"rgbtvg_m3fd"}
MODALITIES=${MODALITIES:-"rgbt"}
EVAL_SETS=${EVAL_SETS:-"test \
test_HW test_RS test_ID test_PL test_IT "}
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-36}
CUDADEVICES=${CUDADEVICES:-0,1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"
LOG_ROOT="./logs/eval/HiVG"
mkdir -p "$LOG_ROOT"

for ds in $DATASETS; do
  for m in $MODALITIES; do
    echo -e "\n==================== [HiVG] DATASET: $ds, MODALITY: $m ==========================="
    LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"
    DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES EVAL_SETS="$EVAL_SETS" \
      stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
  done
done
