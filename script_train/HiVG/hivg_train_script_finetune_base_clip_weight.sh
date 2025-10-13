#!/bin/bash

# ================= 全局参数 =================
export IMGSIZE=224
export BATCHSIZE=16
export MODALITY='rgb'
export CUDADEVICES=1,2,3

# ================= 日志目录 =================
mkdir -p logs/hivg/rgb

# ================= 串行训练四个数据集 =================

echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/HiVG/rgbtvg_flir/hivg_single_dataset_flir_finetune_base_clip_weight.sh 2>&1 | tee logs/hivg/rgb/flir_train.log

echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/HiVG/rgbtvg_m3fd/hivg_single_dataset_m3fd_finetune_base_clip_weight.sh 2>&1 | tee logs/hivg/rgb/m3fd_train.log

echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/HiVG/rgbtvg_mfad/hivg_single_dataset_mfad_finetune_base_clip_weight.sh 2>&1 | tee logs/hivg/rgb/mfad_train.log

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/HiVG/rgbtvg_mixup/hivg_mixup_dataset_finetune_base_clip_weight.sh 2>&1 | tee logs/hivg/rgb/mixup_train.log
