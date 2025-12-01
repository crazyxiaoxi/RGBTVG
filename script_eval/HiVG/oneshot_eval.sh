#!/bin/bash

# HiVG (ViT-B) oneshot evaluation script: 3 datasets × 2 modalities (rgb/ir) × multiple fine-grained test splits
# Default oneshot weights in mixup_pretraining_base/mixup:
#   - rgb/ir:   best_checkpoint.pth
#   - rgbt:     best_checkpoint_rgbt.pth

DATASETS=${DATASETS:-"rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad"}
MODALITIES=${MODALITIES:-"rgb ir"}
EVAL_SETS=${EVAL_SETS:-"test \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY"}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-18}         # Align with batch size in the mixup oneshot training script
CUDADEVICES=${CUDADEVICES:-0,1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"   # Only evaluation, does not go through the retrain logic in single_eval_new
# Only evaluation, does not go through the retrain logic in single_eval_new
LOG_ROOT="$ROOT_DIR/logs/eval/HiVG_oneshot"
mkdir -p "$LOG_ROOT"

for ds in $DATASETS; do
  for m in $MODALITIES; do
    echo -e "\n==================== [HiVG-oneshot] DATASET: $ds, MODALITY: $m ==========================="

    # Select oneshot checkpoint according to modality (can be overridden by environment variables)
    if [[ "$m" == "rgbt" ]]; then
      MODEL_PATH=${HI_B_RGBT_MODEL:-"../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/best_checkpoint_rgbt.pth"}
    else
      MODEL_PATH=${HI_B_RGB_IR_MODEL:-"../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/best_checkpoint.pth"}
    fi

    LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"

    DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES \
    EVAL_SETS="$EVAL_SETS" EVAL_MODEL_PATH="$MODEL_PATH" \
      stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
  done
done
