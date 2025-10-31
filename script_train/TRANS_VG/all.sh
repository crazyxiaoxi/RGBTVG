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
echo "Start TRANS-VG training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

mkdir -p logs/transvg/$MODALITY

DATASET="rgbtvg_flir"
export DATASET
echo "===== Start FLIR training ====="
#如果modality是rgb跳过
if [ "$MODALITY" == "rgb" ]; then
    echo "Skipping FLIR training for rgb modality"
else
    stdbuf -oL -eL bash ./script_train/TRANS_VG/single.sh 2>&1 | tee logs/transvg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_flir.log"
fi

DATASET="rgbtvg_m3fd"
export DATASET
if [ "$MODALITY" == "rgb" ]; then
    echo "Skipping M3FD training for rgb modality"
else
    stdbuf -oL -eL bash ./script_train/TRANS_VG/single.sh 2>&1 | tee logs/transvg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"
fi

DATASET="rgbtvg_mfad"
export DATASET
echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/single.sh 2>&1 | tee logs/transvg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mfad.log"

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/mixup.sh 2>&1 | tee logs/transvg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mixup.log"
