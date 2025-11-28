#!/bin/bash

# HiVG (ViT-B) oneshot 评测脚本：3 数据集 × 3 模态 × 多个细分测试集
# 默认使用 mixup_pretraining_base/mixup 下的 oneshot 权重：
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
BATCHSIZE=${BATCHSIZE:-18}         # 对齐 mixup oneshot 训练脚本中的 batch size
CUDADEVICES=${CUDADEVICES:-0,1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SINGLE_EVAL_SH="$SCRIPT_DIR/single_eval.sh"   # 只做评测，不走 single_eval_new 的 retrain 逻辑
LOG_ROOT="$ROOT_DIR/logs/eval/HiVG_oneshot"
mkdir -p "$LOG_ROOT"

for ds in $DATASETS; do
  for m in $MODALITIES; do
    echo -e "\n==================== [HiVG-oneshot] DATASET: $ds, MODALITY: $m ==========================="

    # 根据模态选择 oneshot 权重（可通过环境变量覆盖）
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
