#!/bin/bash
echo -e "\n\n\n\n\n\n\n==================== mdetr mixup ==========================="

DATA_SET="rgbtvg_mixup"
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-16}
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
EVAL_MODEL_PATH="./output_training/MDETR_clip_${IMGSIZE}_${MODALITY}/$DATA_SET/best_checkpoint.pth"
OUTPUT_DIR="./output_training/MDETR_clip_${IMGSIZE}_${MODALITY}/$DATA_SET"

"${DIST_CMD[@]}" \
    --master_port 28501 \
    mdetr_train.py \
    --model_type CLIP \
    --batch_size $BATCHSIZE \
    --epochs $EPOCHS \
    --lr_bert 0.00001 \
    --aug_crop \
    --aug_scale \
    --aug_translate \
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
    --clip_max_norm 1.0 \
    --data_root $DATA_ROOT \
    --split_root $SPLIT_ROOT \
    --model_name MDETR \
    --modality $MODALITY

evaluate() {
    local eval_set=$1
    "${DIST_CMD[@]}" \
        --master_port 28600 \
        mdetr_eval.py \
        --model_type CLIP \
        --batch_size $BATCHSIZE \
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
        --eval_model $EVAL_MODEL_PATH \
        --eval_set "$eval_set" \
        --data_root $DATA_ROOT \
        --split_root $SPLIT_ROOT \
        --model_name MDETR \
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

    EVAL_MODEL_PATH="./output_training/MDETR_clip_${IMGSIZE}_${MODALITY}/$DATA_SET_TEST/best_checkpoint.pth"
    OUTPUT_DIR="./output_training/MDETR_clip_${IMGSIZE}_${MODALITY}/$DATA_SET_TEST"
    mkdir -p $OUTPUT_DIR

    evaluate "val"
    evaluate "test"
    evaluate "testA"
    evaluate "testB"
    evaluate "testC"
done
