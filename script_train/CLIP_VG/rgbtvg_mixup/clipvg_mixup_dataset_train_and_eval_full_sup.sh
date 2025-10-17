#!/bin/bash
echo -e "\n\n\n\n\n\n\n==================== clipvg mixup ==========================="
# 数据集配置
DATA_SET="rgbtvg_mixup"
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-32}
MODALITY=${MODALITY:-rgbt}
CUDADEVICES=${CUDADEVICES:-0}
EPOCHS=${EPOCHS:-110}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
# 分布式训练配置
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)
# 路径配置
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
EVAL_MODEL_PATH="./output_training/CLIP_VG_${IMGSIZE}_${MODALITY}/$DATA_SET/best_checkpoint.pth"
OUTPUT_DIR="./output_training/CLIP_VG_${IMGSIZE}_${MODALITY}/$DATA_SET"
# 训练参数
TRAIN_ARGS=(--num_workers 4 --modality $MODALITY --batch_size $BATCHSIZE --imsize $IMGSIZE --epochs $EPOCHS --lr 0.00025 --lr_scheduler cosine --aug_crop --aug_scale --aug_translate --vl_hidden_dim 512 --max_query_len 77)
# 评估参数
EVAL_ARGS=( --num_workers 4 --modality $MODALITY --batch_size $BATCHSIZE --imsize $IMGSIZE --max_query_len 77)
# 评估函数
evaluate() {
    local eval_set=$1
    "${DIST_CMD[@]}" \
        --master_port 28876 \
        eval_clip_vg.py \
        "${EVAL_ARGS[@]}" \
        --dataset "$DATA_SET" \
        --data_root "$DATA_ROOT" \
        --split_root "$SPLIT_ROOT" \
        --eval_model "$EVAL_MODEL_PATH" \
        --eval_set "$eval_set" \
        --output_dir "$OUTPUT_DIR"
}
# 训练
"${DIST_CMD[@]}" \
    --master_port 28875 \
    train_clip_vg.py \
    "${TRAIN_ARGS[@]}" \
    --dataset "$DATA_SET" \
    --data_root "$DATA_ROOT" \
    --split_root "$SPLIT_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --sup_type full

# 评估
evaluate "val"
evaluate "test"
evaluate "testA"
evaluate "testB"
evaluate "testC"