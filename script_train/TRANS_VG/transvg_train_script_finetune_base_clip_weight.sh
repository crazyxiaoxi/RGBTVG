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

echo "===== Start TransVG FLIR training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/rgbt_flir/transvg_single_dataset_flir_train_and_eval_full_sup.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/transvg/$MODALITY/flir_train.log

echo "===== Start TransVG M3FD training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/rgbt_m3fd/transvg_single_dataset_m3fd_train_and_eval_full_sup.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/transvg/$MODALITY/m3fd_train.log

echo "===== Start TransVG MFAD training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/rgbt_mfad/transvg_single_dataset_mfad_train_and_eval_full_sup.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/transvg/$MODALITY/mfad_train.log

echo "===== Start TransVG MIXUP training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/rgbtvg_mixup/transvg_mixup_dataset_train_and_eval_full_sup.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/transvg/$MODALITY/mixup_train.log
