#!/bin/bash

# Single-dataset single-modality debug script for HiVG (ViT-B)
# Logic:
# 1) Use the "official result" checkpoint as the --retrain load and train for 1 epoch.
# 2) Reuse the evaluation logic from single_eval.sh to evaluate this 1-epoch checkpoint.

DATA_SET=${DATASET:-rgbtvg_flir}
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
# Official result checkpoint used as the initial model for retraining
OFFICIAL_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/mixup/fixed_best_checkpoint_peft0111.pth"

# "../dataset_and_pretrain_model/result/HiVG_B/HiVG_B_${MODALITY}_$(echo $DATA_SET | sed 's/rgbtvg_//')_best.pth"

# Debug training output directory
OUTPUT_DIR_DEBUG="./output_debug_retrain/HiVG_${IMGSIZE}_${MODALITY}/$DATA_SET"
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
  --vl_hidden_dim 512 \
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
)

EVAL_SETS=${EVAL_SETS:-"test testA testB testC val \
 test_VWL test_WL test_NL test_SL \
 test_NS test_SS \
 test_PO test_HO \
 test_UB test_SU test_RR test_HW test_RS test_ID test_PL test_IT test_TN test_BG test_CP test_MK test_WF \
 test_FY test_RY test_SY test_CY testA testB testC val"}

# Evaluation arguments reused from the original single_eval script
EVAL_MODEL_PATH="$OUTPUT_DIR_DEBUG/best_checkpoint.pth"
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_official_debug/HiVG_${IMGSIZE}_${MODALITY}/$DATA_SET"}
mkdir -p "$OUTPUT_DIR"

EVAL_ARGS=( \
  --modality $MODALITY \
  --batch_size $BATCHSIZE \
  --dataset $DATA_SET \
  --vl_hidden_dim 512 \
  --imsize $IMGSIZE \
  --max_query_len 77 \
  --normalize_before \
  --mixup_pretrain \
  # --enable_adaptive_weights \
  --use_mask_loss \
  --save_hilora_clip \
  # --hi_lora_stage 3 \
  --data_root $DATA_ROOT \
  --split_root $SPLIT_ROOT \
  --output_dir $OUTPUT_DIR \
  --model ViT-B/16 \
)


evaluate() {
  local eval_set=$1
  echo -e "\n>>>> [HiVG-debug] Eval set: $eval_set, model: $EVAL_MODEL_PATH"
  "${DIST_CMD[@]}" \
    --master_port 28873 \
    train_val/hivg_eval.py \
    "${EVAL_ARGS[@]}" \
    --eval_model "$EVAL_MODEL_PATH" \
    --eval_set "$eval_set"
}

# 1) Train for one epoch
echo -e "\n==================== [HiVG-debug] Train 1 epoch from OFFICIAL_MODEL ==========================="
"${DIST_CMD[@]}" --master_port 28870 train_val/hivg_train.py "${TRAIN_ARGS[@]}"

# 2) Then evaluate
for es in $EVAL_SETS; do
  evaluate "$es"
done
