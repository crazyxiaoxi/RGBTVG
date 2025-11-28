#!/bin/bash

IMGSIZE=${1:-224}
BATCHSIZE=${2:-8}
MODALITY=${3:-rgb}
CUDADEVICES=${4:-0,1}
EPOCHS=${5:-110}


# 导出环境变量给第三层脚本
export IMGSIZE
export BATCHSIZE
export MODALITY
export CUDADEVICES
export EPOCHS

echo "Start CLIPVG training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

mkdir -p oneshot_logs/clipvg/$MODALITY

DATASET="rgbtvg_flir"
export DATASET
echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/oneshot.sh 2>&1 | tee oneshot_logs/clipvg/$MODALITY/flir.log

DATASET="rgbtvg_m3fd"
export DATASET
echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/oneshot.sh 2>&1 | tee oneshot_logs/clipvg/$MODALITY/m3fd.log

DATASET="rgbtvg_mfad"
export DATASET
echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/oneshot.sh 2>&1 | tee oneshot_logs/clipvg/$MODALITY/mfad.log
