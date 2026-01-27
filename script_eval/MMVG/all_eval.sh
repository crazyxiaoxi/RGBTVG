#!/bin/bash

# Batch evaluation script: iterate over 3 datasets × 3 modalities (MMVG ViT-B)

DATASETS=${DATASETS:-"rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad"}
MODALITIES=${MODALITIES:-"rgbt"}
EVAL_SETS=${EVAL_SETS:-"test testA testB testC val \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY testA testB testC val"}
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-36}
CUDADEVICES=${CUDADEVICES:-0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"
LOG_ROOT="./logs/eval/MMVG"
mkdir -p "$LOG_ROOT"

for ds in $DATASETS; do
  for m in $MODALITIES; do
    echo -e "\n==================== [MMVG] DATASET: $ds, MODALITY: $m ==========================="
    LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"
    DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES EVAL_SETS="$EVAL_SETS" \
      stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
  done
done
