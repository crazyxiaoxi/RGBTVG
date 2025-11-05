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
echo "Start MMCA CLIP training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

mkdir -p logs/mmca/$MODALITY

DATASET="rgbtvg_m3fd"
export DATASET
echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/single_new.sh 2>&1 | tee logs/mmca/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"
