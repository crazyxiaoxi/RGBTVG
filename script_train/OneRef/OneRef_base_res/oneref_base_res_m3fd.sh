#res和rec有没有 --enable_seg_mask

#!/bin/bash
echo -e "\n\n\n\n\n\n\n==================== oneref training m3fd==========================="

DATA_SET="rgbtvg_m3fd"
IMGSIZE=${IMGSIZE:-224}
BATCHSIZE=${BATCHSIZE:-1}
MODALITY=${MODALITY:-rgb}
CUDADEVICES=${CUDADEVICES:-3}
EPOCHS=${EPOCHS:-1}

NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')

DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES TORCH_USE_CUDA_DSA=1 python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
EVAL_MODEL_PATH="./output_training/ONEREF_base_res_${IMGSIZE}_${MODALITY}/$DATA_SET/best_checkpoint.pth"
OUTPUT_DIR="./output_training/ONEREF_base_res_${IMGSIZE}_${MODALITY}/$DATA_SET"

mkdir -p $OUTPUT_DIR

#==================== TRAIN ====================
"${DIST_CMD[@]}" \
    --master_port 23001 \
    oneref_train.py \
    --modality $MODALITY \
    --num_workers 4 \
    --epochs $EPOCHS \
    --batch_size $BATCHSIZE \
    --lr 0.00025 \
    --lr_scheduler cosine \
    --aug_crop \
    --aug_scale \
    --aug_translate \
    --imsize $IMGSIZE \
    --max_query_len 64 \
    --model beit3_base_patch16_224 \
    --task grounding \
    --dataset $DATA_SET \
    --use_mask_loss \
    --enable_seg_mask \
    --frozen_backbone \
    --sentencepiece_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm\
    --finetune ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3_base_indomain_patch16_224.pth \
    --data_root $DATA_ROOT \
    --split_root $SPLIT_ROOT \
    --output_dir $OUTPUT_DIR


evaluate() {
    local eval_set=$1
    "${DIST_CMD[@]}" \
        --master_port 23002 \
        oneref_eval.py \
        --modality $MODALITY \
        --num_workers 4 \
        --batch_size 2 \
        --imsize $IMGSIZE \
        --max_query_len 64 \
        --model beit3_base_patch16_224 \
        --task grounding \
        --dataset $DATA_SET \
        --use_mask_loss \
        --enable_seg_mask \
        --sentencepiece_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm\
        --finetune ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3_base_indomain_patch16_224.pth \
        --eval_set "$eval_set" \
        --data_root $DATA_ROOT \
        --split_root $SPLIT_ROOT \
        --eval_model $EVAL_MODEL_PATH \
        --output_dir $OUTPUT_DIR
}

# 调用 eval
evaluate "val" 
evaluate "test" 
evaluate "testA" 
evaluate "testB" 
evaluate "testC" 