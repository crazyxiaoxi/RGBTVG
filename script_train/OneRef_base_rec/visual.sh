#!/bin/bash
# ===================== 配置参数 =====================
DATA_SET=${DATASET:-rgbtvg_mfad}
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-1}
MODALITY=${MODALITY:-rgb}
CUDADEVICES=${CUDADEVICES:-0}

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
OUTPUT_DIR="./output_eval/ONEREF_base_${IMGSIZE}_${MODALITY}/$DATA_SET"

mkdir -p $OUTPUT_DIR

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

# 初始化 checkpoint 路径
CHECKPOINT_PATH="$OUTPUT_DIR/best_checkpoint.pth"

# ===================== eval 函数 =====================
evaluate() {
    local eval_set=$1
    local model_path=$2
    "${DIST_CMD[@]}" \
        visualize.py \
        --modality $MODALITY \
        --imsize $IMGSIZE \
        --num_workers 4 \
        --batch_size $BATCHSIZE \
        --max_query_len 64 \
        --model_name OneRef \
        --task grounding \
        --dataset $DATA_SET \
        --use_regress_box \
        --model beit3_base_patch16_224 \
        --sentencepiece_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm \
        --eval_set "$eval_set" \
        --data_root $DATA_ROOT \
        --split_root $SPLIT_ROOT \
        --eval_model $model_path \
        --output_dir $OUTPUT_DIR
}

# ===================== 执行 eval =====================
for split in "val" "test" "testA" "testB" "testC"; do
    evaluate "$split" "$CHECKPOINT_PATH"
done

echo -e "\nEvaluation finished. Final checkpoint: $CHECKPOINT_PATH"
