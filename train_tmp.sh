#!/bin/bash

IMGSIZE=224
BATCHSIZE=36
CUDADEVICES=2,4,6
EPOCHS=120
echo -e "\n\n===================== 启动全部训练与测试 ====================="

# echo -e "\n\n===== 启动 MDETR CLIP 训练与测试 ====="
# MODALITY='rgb'
# bash ./script_train/MDETR_clip/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

# MODALITY='ir'
# bash ./script_train/MDETR_clip/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS



echo -e "\n\n===================== 所有任务已执行完毕 ====================="
