#!/bin/bash
echo -e "\n\n==================== TransVG multi-dataset evaluation (no training) ==========================="

IMGSIZE=${1:-224}
BATCHSIZE=${2:-4}
MODALITY=${3:-rgbt}
CUDADEVICES=${4:-0}
EPOCHS=${EPOCHS:-110}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES TORCH_USE_CUDA_DSA=1 python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"

# ------------------- 数据集列表 -------------------
DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")

# ------------------- 评估函数 -------------------
evaluate() {
    local eval_set=$1
    "${DIST_CMD[@]}" \
        --master_port 28301 \
        transvg_eval.py \
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
        --split_root $SPLIT_ROOT
}

# ------------------- 遍历数据集执行 -------------------
for DATA_SET in "${DATASETS[@]}"; do
    echo -e "\n\n==================== Evaluating dataset: $DATA_SET ==========================="
    OUTPUT_DIR="./output_training/TransVG_${IMGSIZE}_${MODALITY}/${DATA_SET}"
    EVAL_MODEL_PATH="${OUTPUT_DIR}/best_checkpoint.pth"

    mkdir -p $OUTPUT_DIR
    mkdir -p logs/hivg/$MODALITY

    stdbuf -oL -eL bash -c "
        evaluate \"val\"
        evaluate \"test\"
        evaluate \"testA\"
        evaluate \"testB\"
        evaluate \"testC\"
    " 2>&1 | tee logs/hivg/$MODALITY/${DATA_SET}_mixup_val.log
done

echo -e "\n\n==================== All evaluations finished ==========================="
