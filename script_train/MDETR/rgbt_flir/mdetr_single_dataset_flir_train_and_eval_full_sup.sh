#!/bin/bash
echo -e "\n\n\n==================== MDETR FLIR ==========================="

# 数据集配置
DATA_SET="rgbtvg_flir"
IMGSIZE=${IMGSIZE:-640}
BATCHSIZE=${BATCHSIZE:-4}
MODALITY=${MODALITY:-rgb}
CUDADEVICES=${CUDADEVICES:-0}
NPROC_PER_NODE=$(echo "$CUDADEVICES" | tr ',' '\n' | wc -l | awk '{print $1}')

# 分布式训练配置
DIST_CMD=(env CUDA_VISIBLE_DEVICES=$CUDADEVICES python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --use_env)

# 路径配置
DATA_ROOT="../dataset_and_pretrain_model/datasets/VG/image_data"
SPLIT_ROOT="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled"
OUTPUT_DIR="./output_training/MDETR_${IMGSIZE}_${MODALITY}/$DATA_SET"

# 训练
"${DIST_CMD[@]}" \
    --master_port 28885 \
    mdetr_train.py \
    --model_type ResNet \
    --batch_size $BATCHSIZE \
    --epochs 110 \
    --lr_bert 0.00001 \
    --aug_crop \
    --aug_scale \
    --aug_translate \
    --backbone resnet50 \
    --detr_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/Detr/detr-r50.pth \
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
    --split_root $SPLIT_ROOT
