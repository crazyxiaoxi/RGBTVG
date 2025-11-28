#!/bin/bash

# 单数据集单模态：HiVG_L (ViT-L) 调试脚本
# 逻辑：
# 1) 使用 "官方结果" ckpt 作为 --retrain load，训练 1 epoch
# 2) 使用原 single_eval.sh 的评测逻辑，对该 1-epoch 权重进行评测

DATA_SET=${DATASET:-rgbtvg_flir}
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
# 官方结果 ckpt 作为 retrain 初始模型
OFFICIAL_MODEL="../dataset_and_pretrain_model/result/HiVG_L/HiVG_L_${MODALITY}_$(echo $DATA_SET | sed 's/rgbtvg_//')_best.pth"

# debug 训练输出目录
OUTPUT_DIR_DEBUG="./output_debug_retrain/HiVG_L_${IMGSIZE}_${MODALITY}/$DATA_SET"
mkdir -p "$OUTPUT_DIR_DEBUG"

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

TRAIN_ARGS=( \
  --modality $MODALITY \
  --num_workers 4 \
  --epochs ${EPOCHS:-1} \
  --batch_size $BATCHSIZE \
  --lr 0.00025 \
  --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate \
  --vl_hidden_dim 768 \
  --imsize $IMGSIZE \
  --max_query_len 77 \
  --normalize_before \
  --mixup_pretrain \
  --dataset $DATA_SET \
  --use_contrastive_loss \
  --use_rtcc_constrain_loss \
  --use_mask_loss \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --output_dir $OUTPUT_DIR_DEBUG \
  --retrain $OFFICIAL_MODEL \
  --model ViT-L/14 \
)

EVAL_SETS=${EVAL_SETS:-"test \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY"}

# 原 single_eval 的评测参数
EVAL_MODEL_PATH="$OUTPUT_DIR_DEBUG/best_checkpoint.pth"
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_official_debug/HiVG_L_${IMGSIZE}_${MODALITY}/$DATA_SET"}
mkdir -p "$OUTPUT_DIR"

EVAL_ARGS=( \
  --modality $MODALITY \
  --num_workers 2 \
  --batch_size $BATCHSIZE \
  --dataset $DATA_SET \
  --imsize $IMGSIZE \
  --max_query_len 77 \
  --normalize_before \
  --mixup_pretrain \
  --use_mask_loss \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --output_dir $OUTPUT_DIR \
  --vl_hidden_dim 768 \
  --model ViT-L/14 \
)

evaluate() {
  local eval_set=$1
  echo -e "\n>>>> [HiVG_L-debug] Eval set: $eval_set, model: $EVAL_MODEL_PATH"
  "${DIST_CMD[@]}" \
    --master_port 28886 \
    hivg_eval.py \
    "${EVAL_ARGS[@]}" \
    --eval_model "$EVAL_MODEL_PATH" \
    --eval_set "$eval_set"
}

# 1) 先训练一轮
echo -e "\n==================== [HiVG_L-debug] Train 1 epoch from OFFICIAL_MODEL ==========================="
"${DIST_CMD[@]}" --master_port 28880 hivg_train.py "${TRAIN_ARGS[@]}"

# 2) 再评测
for es in $EVAL_SETS; do
  evaluate "$es"
done
