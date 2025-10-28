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
echo "Start ONEREF training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

mkdir -p logs/oneref_base_res/$MODALITY

echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/OneRef/OneRef_base_res/oneref_base_res_flir.sh 2>&1 | tee logs/oneref_base_res/$MODALITY/flir_train.log

echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/OneRef/OneRef_base_res/oneref_base_res_m3fd.sh 2>&1 | tee logs/oneref_base_res/$MODALITY/m3fd_train.log

echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/OneRef/OneRef_base_res/oneref_base_res_mfad.sh 2>&1 | tee logs/oneref_base_res/$MODALITY/mfad_train.log

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/OneRef/OneRef_base_res/oneref_base_res_mixup.sh 2>&1 | tee logs/oneref_base_res/$MODALITY/mixup_train.log
