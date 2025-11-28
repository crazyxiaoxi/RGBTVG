#!/bin/bash

IMGSIZE=${1:-224}
BATCHSIZE=${2:-32}
MODALITY=${3:-rgb}
CUDADEVICES=${4:-0,1}
EPOCHS=${5:-110}

export IMGSIZE
export BATCHSIZE
export MODALITY
export CUDADEVICES
export EPOCHS
echo "Start oneref training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

mkdir -p oneshot_logs/OneRef_large_rec/$MODALITY
if [ "$MODALITY" == "rgb" ]; then
    export PRETRAIN_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/l_rec_224.pth"
elif [ "$MODALITY" == "ir" ]; then
    export PRETRAIN_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/l_rec_224.pth"
elif [ "$MODALITY" == "rgbt" ]; then
    export PRETRAIN_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/l_rec_224.pth"
fi
DATASET="rgbtvg_flir"
export DATASET
echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/OneRef_large_rec/oneshot.sh 2>&1 | tee oneshot_logs/OneRef_large_rec/$MODALITY/flir.log

DATASET="rgbtvg_m3fd"
export DATASET
echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/OneRef_large_rec/oneshot.sh 2>&1 | tee oneshot_logs/OneRef_large_rec/$MODALITY/m3fd.log

DATASET="rgbtvg_mfad"
export DATASET
echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/OneRef_large_rec/oneshot.sh 2>&1 | tee oneshot_logs/OneRef_large_rec/$MODALITY/mfad.log

