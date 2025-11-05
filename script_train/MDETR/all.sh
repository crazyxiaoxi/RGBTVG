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

mkdir -p logs/mdetr_resnet/$MODALITY

# DATASET="rgbtvg_flir"
# export DATASET
# echo "===== Start FLIR training ====="
# if [ "$MODALITY" == "rgb" ]; then
#     echo "Skipping FLIR training for rgb modality"
# else
#     stdbuf -oL -eL bash ./script_train/MDETR/single.sh 2>&1 | tee logs/mdetr_resnet/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_flir.log"
# fi


# DATASET="rgbtvg_m3fd"
# export DATASET
# echo "===== Start M3FD training ====="
# stdbuf -oL -eL bash ./script_train/MDETR/single.sh 2>&1 | tee logs/mdetr_resnet/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"

# DATASET="rgbtvg_mfad"
# export DATASET
# echo "===== Start MFAD training ====="
# stdbuf -oL -eL bash ./script_train/MDETR/single.sh 2>&1 | tee logs/mdetr_resnet/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mfad.log"

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/MDETR/mixup.sh 2>&1 | tee logs/mdetr_resnet/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mixup.log"
