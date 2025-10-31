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

DATASET="rgbtvg_flir"
export DATASET
if [ "$MODALITY" == "rgb" ]; then
    echo "Skipping FLIR training for rgb modality"
else
    echo "===== Start FLIR training ====="
    stdbuf -oL -eL bash ./script_train/MMCA_VG/single.sh 2>&1 | tee logs/mmca/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_flir.log"
fi

DATASET="rgbtvg_m3fd"
export DATASET
echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/single.sh 2>&1 | tee logs/mmca/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"


DATASET="rgbtvg_mfad"
export DATASET
if [ "$MODALITY" == "rgb" ]; then
    echo "Skipping MFAD training for rgb modality"
else
    echo "===== Start MFAD training ====="
    stdbuf -oL -eL bash ./script_train/MMCA_VG/single.sh 2>&1 | tee logs/mmca/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mfad.log"
fi

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/mixup.sh 2>&1 | tee logs/mmca/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mixup.log"
