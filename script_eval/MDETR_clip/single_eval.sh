#!/bin/bash

# Single-dataset single-modality evaluation script for MDETR CLIP
DATA_SET=${DATASET:-rgbtvg_flir}

echo -e "\n\n\n\n\n\n\n==================== MDETR_clip single eval dataset: $DATA_SET ==========================="
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-16}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}

# Default evaluation split list (kept consistent with CLIP_VG)
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
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-"../dataset_and_pretrain_model/result/MDETR_clip/MDETR_${IMGSIZE}_clip_${MODALITY}_$(echo $DATA_SET | sed 's/rgbtvg_//')_best.pth"}
# eval_official is used for logs and results inside the Python script
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_official/MDETR_clip_${IMGSIZE}_${MODALITY}/$DATA_SET"}

EVAL_ARGS=( \
  --model_type CLIP \
  --batch_size $BATCHSIZE \
  --imsize $IMGSIZE \
  --backbone resnet50 \
  --bert_enc_num 12 \
  --detr_enc_num 6 \
  --dataset $DATA_SET \
  --max_query_len 40 \
  --output_dir $OUTPUT_DIR \
  --stages 3 \
  --vl_fusion_enc_layers 3 \
  --uniform_learnable True \
  --in_points 36 \
  --lr 1e-4 \
  --different_transformer True \
  --lr_drop 60 \
  --vl_dec_layers 1 \
  --vl_enc_layers 1 \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --model_name MDETR \
  --modality $MODALITY \
)

evaluate() {
  local eval_set=$1
  echo -e "\n>>>> [MDETR_clip] Eval set: $eval_set, model: $EVAL_MODEL_PATH"
  "${DIST_CMD[@]}" \
    --master_port 28600 \
    mdetr_eval.py \
    "${EVAL_ARGS[@]}" \
    --eval_model "$EVAL_MODEL_PATH" \
    --eval_set "$eval_set"
}

for es in $EVAL_SETS; do
  evaluate "$es"
done
