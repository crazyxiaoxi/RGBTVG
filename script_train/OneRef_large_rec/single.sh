#!/bin/bash
# ===================== 配置参数 =====================
DATA_SET=${DATASET:-rgbtvg_flir}
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-1}
MODALITY=${MODALITY:-rgb}
CUDADEVICES=${CUDADEVICES:-3}
TOTAL_EPOCHS=${EPOCHS:-12}   # 总轮数
PRETRAIN_MODEL=${PRETRAIN_MODEL:-""}

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
OUTPUT_DIR="./output_training/ONEREF_large_rec_${IMGSIZE}_${MODALITY}/$DATA_SET"

mkdir -p $OUTPUT_DIR

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES TORCH_USE_CUDA_DSA=1 python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

# ===================== 训练轮次配置 =====================
EPOCH_FROZEN=$(($TOTAL_EPOCHS / 6))
EPOCH_MASK=$(($TOTAL_EPOCHS / 3))

# 初始化 checkpoint 路径
CHECKPOINT_PATH=$PRETRAIN_MODEL

# ===================== eval 函数 =====================
evaluate() {
    local eval_set=$1
    local model_path=$2
    "${DIST_CMD[@]}" \
        --master_port $((35000 + ROUND)) \
        oneref_eval.py \
        --modality $MODALITY \
        --imsize $IMGSIZE \
        --num_workers 4 \
        --batch_size $BATCHSIZE \
        --max_query_len 64 \
        --model beit3_large_patch16_224 \
        --task grounding \
        --dataset $DATA_SET \
        --use_regress_box \
        --sentencepiece_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm \
        --finetune $model_path \
        --eval_set "$eval_set" \
        --data_root $DATA_ROOT \
        --split_root $SPLIT_ROOT \
        --eval_model $model_path \
        --output_dir $OUTPUT_DIR
}

===================== 训练轮次 =====================
for ROUND in {1..2}; do
    echo -e "\n\n========== Training Round $ROUND =========="

    # ---- 冻结 backbone 阶段 ----
    "${DIST_CMD[@]}" \
        --master_port $((33000 + ROUND)) \
        oneref_train.py \
        --modality $MODALITY \
        --imsize $IMGSIZE \
        --retrain $CHECKPOINT_PATH \
        --num_workers 4 \
        --epochs $EPOCH_FROZEN \
        --batch_size $(($BATCHSIZE / 1)) \
        --lr 0.00025  \
        --lr_scheduler cosine \
        --aug_crop \
        --aug_scale \
        --aug_translate \
        --max_query_len 64 \
        --model beit3_large_patch16_224 \
        --task grounding \
        --dataset $DATA_SET \
        --use_regress_box \
        --frozen_backbone \
        --sentencepiece_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm \
        --finetune $CHECKPOINT_PATH \
        --data_root $DATA_ROOT \
        --split_root $SPLIT_ROOT \
        --output_dir $OUTPUT_DIR

    # 更新 checkpoint 为最新输出
    CHECKPOINT_PATH="$OUTPUT_DIR/best_checkpoint.pth"

    # ---- 使用 box mask constraints 阶段 ----
    "${DIST_CMD[@]}" \
        --master_port $((34000 + ROUND)) \
        oneref_train.py \
        --modality $MODALITY \
        --imsize $IMGSIZE \
        --retrain $CHECKPOINT_PATH \
        --num_workers 4 \
        --epochs $EPOCH_MASK \
        --batch_size $(($BATCHSIZE / 40)) \
        --lr 0.00003  \
        --lr_scheduler cosine \
        --aug_crop \
        --aug_scale \
        --aug_translate \
        --max_query_len 64 \
        --model beit3_large_patch16_224 \
        --task grounding \
        --dataset $DATA_SET \
        --use_regress_box \
        --use_box_mask_constraints \
        --sentencepiece_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm \
        --finetune $CHECKPOINT_PATH \
        --data_root $DATA_ROOT \
        --split_root $SPLIT_ROOT \
        --output_dir $OUTPUT_DIR

    # 更新 checkpoint 为最新输出
    CHECKPOINT_PATH="$OUTPUT_DIR/best_checkpoint.pth"

    # ---- eval ----

done
CHECKPOINT_PATH="$OUTPUT_DIR/best_checkpoint.pth"
for split in "val" "test" "testA" "testB" "testC"; do
    evaluate "$split" "$CHECKPOINT_PATH"
done
echo -e "\nTraining + Evaluation finished. Final checkpoint: $CHECKPOINT_PATH"

