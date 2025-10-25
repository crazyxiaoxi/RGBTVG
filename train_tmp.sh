#!/bin/bash

IMGSIZE=224
BATCHSIZE=1
CUDADEVICES=2
MODALITY='rgb'
EPOCHS=1
echo -e "\n\n===================== 启动全部训练与测试 ====================="

# echo -e "\n\n===== 启动 CLIPVG 训练与测试 ====="
bash ./script_train/CLIP_VG/clipvg_train_script_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

echo -e "\n\n===== 启动 MDETR 训练与测试 ====="
# bash ./script_train/MDETR_clip/mdetr_train_script_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS
# bash ./script_train/MDETR/mdetr_train_script_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

echo -e "\n\n===== 启动 MMCA 训练与测试 ====="
# bash ./script_train/MMCA_VG/mmca_train_script_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

echo -e "\n\n===== 启动 TRANSVG 训练与测试 ====="
# bash ./script_train/TRANS_VG/transvg_train_script_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

echo -e "\n\n===== 启动 HIVG 训练与测试 ====="
# bash ./script_train/HiVG/hivg_train_script_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

# ---- 结束 ----
echo -e "\n\n===================== 所有任务已执行完毕 ====================="
