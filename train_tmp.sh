#!/bin/bash

IMGSIZE=224
BATCHSIZE=36
CUDADEVICES=0,1
EPOCHS=120
echo -e "\n\n===================== 启动全部训练与测试 ====================="


# MODALITY='rgb'
# bash ./script_train/OneRef_base_rec/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

# MODALITY='ir'
# bash ./script_train/OneRef_base_rec/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

echo -e "\n\n===== 启动 MMCA 训练与测试 ====="
# MODALITY='rgb'
# bash ./script_train/MMCA_VG/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

MODALITY='ir'
bash ./script_train/MMCA_VG/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

MODALITY='rgbt'
bash ./script_train/MMCA_VG/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

# echo -e "\n\n===================== 所有任务已执行完毕 ====================="
