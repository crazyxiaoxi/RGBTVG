#!/bin/bash

# 单数据集单模态：MMCA_VG 评测脚本
DATA_SET=${DATASET:-rgbtvg_flir}

echo -e "\n\n\n\n\n\n\n==================== MMCA_VG single eval dataset: $DATA_SET ==========================="
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-8}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}

EVAL_SETS=${EVAL_SETS:-"test \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY"}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-"../dataset_and_pretrain_model/result/MMCA/MMCA_${IMGSIZE}_${MODALITY}_$(echo $DATA_SET | sed 's/rgbtvg_//')_best.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_official/MMCA_${IMGSIZE}_${MODALITY}/$DATA_SET"}

EVAL_ARGS=( \
  --batch_size $BATCHSIZE \
  --imsize $IMGSIZE \
  --num_workers 4 \
  --bert_enc_num 12 \
  --detr_enc_num 6 \
  --backbone resnet50 \
  --dataset $DATA_SET \
  --max_query_len 20 \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --modality $MODALITY \
  --output_dir $OUTPUT_DIR \
)

evaluate() {
  local eval_set=$1
  echo -e "\n>>>> [MMCA_VG] Eval set: $eval_set, model: $EVAL_MODEL_PATH"
  "${DIST_CMD[@]}" \
    --master_port 25401 \
    mmca_eval.py \
    "${EVAL_ARGS[@]}" \
    --eval_set "$eval_set" \
    --eval_model "$EVAL_MODEL_PATH"
}

for es in $EVAL_SETS; do
  evaluate "$es"
done
