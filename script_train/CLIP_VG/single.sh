#!/bin/bash

# Dataset configuration
DATA_SET=${DATASET:-rgbtvg_flir}

echo -e "\n\n\n\n\n\n\n==================== clipvg single dataset: $DATA_SET ==========================="
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgb}
CUDADEVICES=${CUDADEVICES:-0}
EPOCHS=${EPOCHS:-110}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
# Distributed training configuration
echo "use imgsize $IMGSIZE batchsize $BATCHSIZE modality $MODALITY cudadevices $CUDADEVICES nproc_per_node $NPROC_PER_NODE"
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)
# Path configuration
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
EVAL_MODEL_PATH="./output_training/CLIP_VG_${IMGSIZE}_${MODALITY}/$DATA_SET/best_checkpoint.pth"
OUTPUT_DIR="./output_training/CLIP_VG_${IMGSIZE}_${MODALITY}/$DATA_SET"
# Training arguments
TRAIN_ARGS=(--num_workers 4 --modality $MODALITY --batch_size $BATCHSIZE --imsize $IMGSIZE --epochs $EPOCHS --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --vl_hidden_dim 512 --max_query_len 77)
# Evaluation arguments
EVAL_ARGS=( --num_workers 4 --modality $MODALITY --batch_size $BATCHSIZE --imsize $IMGSIZE --max_query_len 77)
# Evaluation function
evaluate() {
    local eval_set=$1
    "${DIST_CMD[@]}" \
        --master_port 28881 \
        eval_clip_vg.py \
        "${EVAL_ARGS[@]}" \
        --dataset "$DATA_SET" \
        --data_root "$DATA_ROOT" \
        --split_root "$SPLIT_ROOT" \
        --eval_model "$EVAL_MODEL_PATH" \
        --eval_set "$eval_set" \
        --output_dir "$OUTPUT_DIR"
}
# Training
"${DIST_CMD[@]}" \
    --master_port 28887 \
    train_clip_vg.py \
    "${TRAIN_ARGS[@]}" \
    --dataset "$DATA_SET" \
    --data_root "$DATA_ROOT" \
    --split_root "$SPLIT_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --sup_type full

# Evaluation
evaluate "val"
evaluate "test"
evaluate "testA"
evaluate "testB"
evaluate "testC"