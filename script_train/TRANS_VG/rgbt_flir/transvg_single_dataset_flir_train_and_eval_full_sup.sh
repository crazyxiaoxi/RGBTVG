#!/bin/bash
echo -e "\n\n\n\n\n\n\n==================== TransVG FLIR ==========================="

# 环境变量配置
DATA_SET="rgbtvg_flir"
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-4}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-3}
NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')

# 路径配置
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
PRETRAIN_DETR="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/Detr/detr-r50.pth"
OUTPUT_DIR="./output_training/TransVG_${IMGSIZE}_${MODALITY}/$DATA_SET"
EVAL_MODEL_PATH="${OUTPUT_DIR}/best_checkpoint.pth"

# 分布式训练命令
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

# 评估函数
evaluate() {
    local eval_set=$1
    "${DIST_CMD[@]}" \
        transvg_eval.py \
        --batch_size 32 \
        --num_workers 4 \
        --bert_enc_num 12 \
        --detr_enc_num 6 \
        --backbone resnet50 \
        --dataset "$DATA_SET" \
        --max_query_len 20 \
        --eval_model "$EVAL_MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --eval_set "$eval_set" \
        --data_root "$DATA_ROOT" \
        --split_root "$SPLIT_ROOT"
}

# 训练
"${DIST_CMD[@]}" \
    --master_port 28700 \
    transvg_train.py \
    --batch_size 4 \
    --lr_bert 0.00001 \
    --aug_crop \
    --aug_scale \
    --aug_translate \
    --backbone resnet50 \
    --detr_model "$PRETRAIN_DETR" \
    --bert_enc_num 12 \
    --detr_enc_num 6 \
    --dataset "$DATA_SET" \
    --data_root "$DATA_ROOT" \
    --split_root "$SPLIT_ROOT" \
    --max_query_len 20 \
    --output_dir "$OUTPUT_DIR" \
    --epochs 110 \
    --lr_drop 60

# 评估
evaluate "val"
evaluate "test"
evaluate "testA"
evaluate "testB"
evaluate "testC"
