#!/bin/bash

# HiVG_L (ViT-L) oneshot evaluation script: 3 datasets × 2 modalities (rgb/ir) × multiple fine-grained test splits
# Default oneshot weights in mixup_pretraining_large/mixup:
#   - rgb/ir:   best_checkpoint.pth
#   - rgbt:     best_checkpoint_rgbt.pth

DATASETS=${DATASETS:-"rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad"}
MODALITIES=${MODALITIES:-"rgb ir"}
EVAL_SETS=${EVAL_SETS:-"test testA testB testC val \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY testA testB testC val"}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-8}
CUDADEVICES=${CUDADEVICES:-0,1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"
LOG_ROOT="$ROOT_DIR/logs/eval/HiVG_L_oneshot"
mkdir -p "$LOG_ROOT"

for ds in $DATASETS; do
  for m in $MODALITIES; do
    echo -e "\n==================== [HiVG_L-oneshot] DATASET: $ds, MODALITY: $m ==========================="

    if [[ "$m" == "rgbt" ]]; then
      MODEL_PATH=${HI_L_RGBT_MODEL:-"../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_large/mixup/best_checkpoint_rgbt.pth"}
    else
      MODEL_PATH=${HI_L_RGB_IR_MODEL:-"../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_large/mixup/best_checkpoint.pth"}
    fi

    LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"

    DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES \
    EVAL_SETS="$EVAL_SETS" EVAL_MODEL_PATH="$MODEL_PATH" \
      stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
  done
done
