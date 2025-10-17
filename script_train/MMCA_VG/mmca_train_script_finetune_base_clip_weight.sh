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

echo "Start MMCA training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

mkdir -p logs/mmca/$MODALITY

echo "===== Start mmca FLIR training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/rgbt_flir/mmca_single_dataset_flir_train_and_eval_full_sup.sh 2>&1 | tee logs/mmca/$MODALITY/flir_train.log

echo "===== Start mmca M3FD training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/rgbt_m3fd/mmca_single_dataset_m3fd_train_and_eval_full_sup.sh 2>&1 | tee logs/mmca/$MODALITY/m3fd_train.log

echo "===== Start mmca MFAD training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/rgbt_mfad/mmca_single_dataset_mfad_train_and_eval_full_sup.sh 2>&1 | tee logs/mmca/$MODALITY/mfad_train.log

echo "===== Start mmca MIXUP training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/rgbtvg_mixup/mmca_mixup_dataset_train_and_eval_full_sup.sh 2>&1 | tee logs/mmca/$MODALITY/mixup_train.log
