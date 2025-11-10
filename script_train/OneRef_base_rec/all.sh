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
echo "Start oneref training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

mkdir -p logs/OneRef_base_rec/$MODALITY
if [ "$MODALITY" == "rgb" ]; then
    export PRETRAIN_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/b_rec_224.pth"
elif [ "$MODALITY" == "ir" ]; then
    export PRETRAIN_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/b_rec_224.pth"
elif [ "$MODALITY" == "rgbt" ]; then
    export PRETRAIN_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/b_rec_224.pth"
fi
DATASET="rgbtvg_flir"
export DATASET
echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/OneRef_base_rec/single.sh 2>&1 | tee logs/OneRef_base_rec/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_flir.log"

DATASET="rgbtvg_m3fd"
export DATASET
echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/OneRef_base_rec/single.sh 2>&1 | tee logs/OneRef_base_rec/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"

DATASET="rgbtvg_mfad"
export DATASET
echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/OneRef_base_rec/single.sh 2>&1 | tee logs/OneRef_base_rec/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mfad.log"


DATASET="rgbtvg_mixup"
export DATASET
echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/OneRef_base_rec/mixup.sh 2>&1 | tee logs/OneRef_base_rec/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mixup.log"
