#!/bin/bash

# ============ 全局训练参数 ============
export IMGSIZE=640
export BATCHSIZE=8
MODALITY='rgb'
export CUDADEVICES=3

# ============ 日志目录 ============
mkdir -p logs/mdetr/rgb

# ============ 顺序训练四个数据集 ============
echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/MDETR/rgbt_flir/mdetr_single_dataset_flir_train_and_eval_full_sup.sh 2>&1 | tee logs/mdetr/rgb/flir_train.log

echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/MDETR/rgbt_m3fd/mdetr_single_dataset_m3fd_train_and_eval_full_sup.sh 2>&1 | tee logs/mdetr/rgb/m3fd_train.log

echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/MDETR/rgbt_mfad/mdetr_single_dataset_mfad_train_and_eval_full_sup.sh 2>&1 | tee logs/mdetr/rgb/mfad_train.log

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/MDETR/rgbtvg_mixup/mdetr_mixup_dataset_train_and_eval_full_sup.sh 2>&1 | tee logs/mdetr/rgb/mixup_train.log
