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

DATASET="rgbtvg_flir"
export DATASET
echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/single_dataset_train_val.sh 2>&1 | tee logs/clipvg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_flir.log"

DATASET="rgbtvg_m3fd"
export DATASET
echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/single_dataset_train_val.sh 2>&1 | tee logs/clipvg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"

DATASET="rgbtvg_mfad"
export DATASET
echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/single_dataset_train_val.sh 2>&1 | tee logs/clipvg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mfad.log"

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/mixup_dataset_train_val.sh 2>&1 | tee logs/clipvg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mixup.log"
