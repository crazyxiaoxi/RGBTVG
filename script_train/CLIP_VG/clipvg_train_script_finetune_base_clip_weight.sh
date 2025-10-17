#!/bin/bash

IMGSIZE=${1:-224}
BATCHSIZE=${2:-8}
MODALITY=${3:-rgb}
CUDADEVICES=${4:-1,2,3}
EPOCHS=${5:-110}

# 导出环境变量给第三层脚本
export IMGSIZE
export BATCHSIZE
export MODALITY
export CUDADEVICES
export EPOCHS

echo "Start CLIPVG training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

mkdir -p logs/clipvg/$MODALITY

echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/rgbt_flir/clipvg_single_dataset_flir_train_and_eval_full_sup.sh 2>&1 | tee logs/clipvg/$MODALITY/flir_train.log

echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/rgbt_m3fd/clipvg_single_dataset_m3fd_train_and_eval_full_sup.sh 2>&1 | tee logs/clipvg/$MODALITY/m3fd_train.log

echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/rgbt_mfad/clipvg_single_dataset_mfad_train_and_eval_full_sup.sh 2>&1 | tee logs/clipvg/$MODALITY/mfad_train.log

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/rgbtvg_mixup/clipvg_mixup_dataset_train_and_eval_full_sup.sh 2>&1 | tee logs/clipvg/$MODALITY/mixup_train.log
