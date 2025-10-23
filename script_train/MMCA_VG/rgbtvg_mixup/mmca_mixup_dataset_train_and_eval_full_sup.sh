#!/bin/bash
echo -e "\n\n\n\n\n\n\n==================== mmca mixup ==========================="

DATA_SET="rgbtvg_mixup"
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-8}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}
EPOCHS=${EPOCHS:-110}

if [[ "$MODALITY" == "ir" || "$MODALITY" == "rgb" ]]; then
    echo "MODALITY is '$MODALITY', script will not run."
    exit 0
fi

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')

DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
EVAL_MODEL_PATH="./output_training/MMCA_$MODALITY/$DATA_SET/best_checkpoint.pth"
OUTPUT_DIR="./output_training/MMCA_$MODALITY/$DATA_SET"

mkdir -p $OUTPUT_DIR

# ==================== TRAIN ====================
"${DIST_CMD[@]}" \
    --master_port 28400 \
    mmca_train.py \
    --batch_size $BATCHSIZE \
    --lr_bert 0.00001 \
    --aug_crop \
    --aug_scale \
    --aug_translate \
    --backbone resnet50 \
    --detr_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/Detr/detr-r50.pth \
    --bert_enc_num 12 \
    --detr_enc_num 6 \
    --dataset $DATA_SET \
    --max_query_len 20 \
    --output_dir $OUTPUT_DIR \
    --data_root $DATA_ROOT \
    --split_root $SPLIT_ROOT \
    --modality $MODALITY \
    --epochs $EPOCHS \
    --lr_drop 60

# ==================== EVALUATE ====================
evaluate() {
    local eval_set=$1
    "${DIST_CMD[@]}" \
        --master_port 28401 \
        mmca_eval.py \
        --batch_size 32 \
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
        --split_root $SPLIT_ROOT \
        --modality $MODALITY
}

evaluate "val"
evaluate "test"
evaluate "testA"
evaluate "testB"
evaluate "testC"


DATA_SET_TEST="rgbtvg_mixup"
OTHER_DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")

for DATA_SET in "${OTHER_DATASETS[@]}"; do
    echo -e "\n\n==================== Evaluating dataset: $DATA_SET ==========================="

    EVAL_MODEL_PATH="./output_training/MMCA_$MODALITY/$DATA_SET_TEST/best_checkpoint.pth"
    OUTPUT_DIR="./output_training/MMCA_$MODALITY/$DATA_SET_TEST"
    mkdir -p $OUTPUT_DIR

    evaluate "val"
    evaluate "test"
    evaluate "testA"
    evaluate "testB"
    evaluate "testC"
done
