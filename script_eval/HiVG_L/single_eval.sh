#!/bin/bash

# Single-dataset single-modality evaluation script for HiVG_L (ViT-L)
DATA_SET=${DATASET:-rgbtvg_flir}

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}

EVAL_SETS=${EVAL_SETS:-"test \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY"}

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
# Official checkpoint path: ../dataset_and_pretrain_model/result/HiVG_L/HiVG_L_<mod>_<ds>_best.pth
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-"../dataset_and_pretrain_model/result/HiVG_L/HiVG_L_${MODALITY}_$(echo $DATA_SET | sed 's/rgbtvg_//')_best.pth"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_official/HiVG_L_${IMGSIZE}_${MODALITY}/$DATA_SET"}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

EVAL_ARGS=( \
  --modality $MODALITY \
  --batch_size $BATCHSIZE \
  --dataset $DATA_SET \
  --vl_hidden_dim 768 \
  --imsize $IMGSIZE \
  --max_query_len 77 \
  --normalize_before \
  --mixup_pretrain \
  --use_mask_loss \
  --save_hilora_clip \
  # --hi_lora_stage 3 \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --output_dir $OUTPUT_DIR \
  --model ViT-L/14 \
)

evaluate() {
  local eval_set=$1
  echo -e "\n>>>> [HiVG_L] Eval set: $eval_set, model: $EVAL_MODEL_PATH"
  "${DIST_CMD[@]}" \
    --master_port 28886 \
    hivg_eval.py \
    "${EVAL_ARGS[@]}" \
    --eval_model "$EVAL_MODEL_PATH" \
    --eval_set "$eval_set"
}

for es in $EVAL_SETS; do
  evaluate "$es"
done
