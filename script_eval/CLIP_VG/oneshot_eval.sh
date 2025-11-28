#!/bin/bash

# CLIP_VG oneshot 评测脚本：3 数据集 × 3 模态 × 多个细分测试集
# 默认使用 ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/clipvg/best_checkpoint.pth
# （路径与 script_train/CLIP_VG/oneshot.sh 中的 EVAL_MODEL_PATH 保持一致）

DATASETS=${DATASETS:-"rgbtvg_flir rgbtvg_m3fd rgbtvg_mfad"}
MODALITIES=${MODALITIES:-"rgb ir"}
EVAL_SETS=${EVAL_SETS:-"test \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY"}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-36}      # 对齐 CLIP_VG/single_eval.sh 默认 batch size
CUDADEVICES=${CUDADEVICES:-0,1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"
LOG_ROOT="$ROOT_DIR/logs/eval/CLIP_VG_oneshot"
mkdir -p "$LOG_ROOT"

# 默认 oneshot ckpt，可通过 CLIPVG_ONESHOT_PATH 覆盖
DEFAULT_MODEL_PATH="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/clipvg/best_checkpoint.pth"
MODEL_PATH=${CLIPVG_ONESHOT_PATH:-$DEFAULT_MODEL_PATH}

for ds in $DATASETS; do
  for m in $MODALITIES; do
    echo -e "\n==================== [CLIP_VG-oneshot] DATASET: $ds, MODALITY: $m ==========================="

    LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"

    DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES \
    EVAL_SETS="$EVAL_SETS" EVAL_MODEL_PATH="$MODEL_PATH" \
      stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
  done
done
