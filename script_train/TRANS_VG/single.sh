#!/bin/bash
DATA_SET=${DATASET:-rgbtvg_flir}
echo -e "\n\n\n\n\n\n\n==================== transvg single dataset: $DATA_SET ==========================="

IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-4}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-3}
EPOCHS=${EPOCHS:-110}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')

DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES TORCH_USE_CUDA_DSA=1 python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
EVAL_MODEL_PATH="./output_training/TransVG_${IMGSIZE}_${MODALITY}/$DATA_SET/best_checkpoint.pth"
OUTPUT_DIR="./output_training/TransVG_${IMGSIZE}_${MODALITY}/$DATA_SET"

mkdir -p $OUTPUT_DIR

# ==================== TRAIN ====================
"${DIST_CMD[@]}" \
    --master_port 28300 \
    train_val/transvg_train.py \
    --batch_size $BATCHSIZE \
    --imsize $IMGSIZE \
    --lr_bert 0.00001 \
    --aug_crop \
    --aug_scale \
    --aug_translate \
    --backbone resnet50 \
    --detr_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/Detr/detr-r50.pth \
    --bert_enc_num 12 \
    --detr_enc_num 6 \
    --dataset $DATA_SET \
    --data_root $DATA_ROOT \
    --split_root $SPLIT_ROOT \
    --max_query_len 20 \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --lr_drop 60

# ==================== EVALUATE ====================
evaluate() {
    local eval_set=$1
    "${DIST_CMD[@]}" \
        --master_port 28301 \
        train_val/transvg_eval.py \
        --imsize $IMGSIZE \
        --batch_size $BATCHSIZE \
        --num_workers 4 \
        --bert_enc_num 12 \
        --detr_enc_num 6 \
        --backbone resnet50 \
        --dataset $DATA_SET \
        --max_query_len 20 \
        --eval_set "$eval_set" \
        --eval_model $EVAL_MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --data_root $DATA_ROOT \
        --split_root $SPLIT_ROOT
}

evaluate "val"
evaluate "test"
evaluate "testA"
evaluate "testB"
evaluate "testC"
