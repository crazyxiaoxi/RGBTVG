#!/bin/bash

# OneRef_base_rec oneshot 评测脚本：3 数据集 × 3 模态 × 多个细分测试集
# 默认使用 oneref/b_rec_224.pth 作为 oneshot 权重（与 oneshot_all.sh 保持一致）。

DATASETS=${DATASETS:-"rgbtvg_m3fd rgbtvg_mfad"}
MODALITIES=${MODALITIES:-"rgb ir"}
EVAL_SETS=${EVAL_SETS:-"test \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY"}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-18}   # 与 OneRef_base_rec/all_eval.sh 默认一致
CUDADEVICES=${CUDADEVICES:-0,1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"
LOG_ROOT="./logs/eval/OneRef_base_rec_oneshot"
mkdir -p "$LOG_ROOT"

# oneshot 权重（可通过 ONEREF_BASE_ONESHOT_PATH 覆盖）
DEFAULT_MODEL_PATH="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/b_rec_224.pth"
MODEL_PATH=${ONEREF_BASE_ONESHOT_PATH:-$DEFAULT_MODEL_PATH}

for ds in $DATASETS; do
  for m in $MODALITIES; do
    echo -e "\n==================== [OneRef_base_rec-oneshot] DATASET: $ds, MODALITY: $m ==========================="

    LOG_FILE="$LOG_ROOT/${IMGSIZE}_${m}_${ds}.log"

    DATASET=$ds MODALITY=$m IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDADEVICES=$CUDADEVICES \
    EVAL_SETS="$EVAL_SETS" EVAL_MODEL_PATH="$MODEL_PATH" \
      stdbuf -oL -eL bash "$SINGLE_EVAL_SH" 2>&1 | tee "$LOG_FILE"
  done
done
