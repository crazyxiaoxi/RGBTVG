#!/bin/bash

IMGSIZE=${1:-224}
BATCHSIZE=${2:-32}
MODALITY=${3:-rgbt}
CUDADEVICES=${4:-0}
EPOCHS=${5:-110}

export IMGSIZE
export BATCHSIZE
export MODALITY
export CUDADEVICES
export EPOCHS
echo "Start MDETR training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

mkdir -p logs/mdetr_clip/$MODALITY

echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/MDETR_clip/rgbt_flir/mdetr_single_dataset_flir_train_and_eval_full_sup.sh 2>&1 | tee logs/mdetr_clip/$MODALITY/flir_train.log

echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/MDETR_clip/rgbt_m3fd/mdetr_single_dataset_m3fd_train_and_eval_full_sup.sh 2>&1 | tee logs/mdetr_clip/$MODALITY/m3fd_train.log

echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/MDETR_clip/rgbt_mfad/mdetr_single_dataset_mfad_train_and_eval_full_sup.sh 2>&1 | tee logs/mdetr_clip/$MODALITY/mfad_train.log

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/MDETR_clip/rgbtvg_mixup/mdetr_mixup_dataset_train_and_eval_full_sup.sh 2>&1 | tee logs/mdetr_clip/$MODALITY/mixup_train.log
