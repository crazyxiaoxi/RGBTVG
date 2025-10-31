#!/bin/bash

IMGSIZE=224
BATCHSIZE=36
CUDADEVICES=0,1,3
EPOCHS=120
echo -e "\n\n===================== 启动全部训练与测试 ====================="

# echo -e "\n\n===== 启动 HIVG 训练与测试 ====="
# MODALITY='rgbt'
# bash ./script_train/HiVG/hivg_train_script_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

# MODALITY='ir'
# bash ./script_train/HiVG/hivg_train_script_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

# echo -e "\n\n===== 启动 OneRef base res训练与测试 ====="

# MODALITY='rgb'
# bash ./script_train/OneRef_base_res/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

# MODALITY='ir'
# bash ./script_train/OneRef_base_res/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

# echo -e "\n\n===== 启动 Transvg 训练与测试 ====="
MODALITY='rgb'
bash ./script_train/TRANS_VG/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

MODALITY='ir'
bash ./script_train/TRANS_VG/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

MODALITY='rgbt'
bash ./script_train/TRANS_VG/all.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES $EPOCHS

echo -e "\n\n===================== 所有任务已执行完毕 ====================="
