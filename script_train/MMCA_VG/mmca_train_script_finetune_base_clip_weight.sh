#!/bin/bash

export IMGSIZE=224
export BATCHSIZE=16
MODALITY='rgb'
export CUDADEVICES=1,2,3

mkdir -p logs/mmca/rgb

echo "===== Start mmca FLIR training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/rgbt_flir/mmca_single_dataset_flir_train_and_eval_full_sup.sh 2>&1 | tee logs/mmca/rgb/flir_train.log

echo "===== Start mmca M3FD training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/rgbt_m3fd/mmca_single_dataset_m3fd_train_and_eval_full_sup.sh 2>&1 | tee logs/mmca/rgb/m3fd_train.log

echo "===== Start mmca MFAD training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/rgbt_mfad/mmca_single_dataset_mfad_train_and_eval_full_sup.sh 2>&1 | tee logs/mmca/rgb/mfad_train.log

echo "===== Start mmca MIXUP training ====="
stdbuf -oL -eL bash ./script_train/MMCA_VG/rgbtvg_mixup/mmca_mixup_dataset_train_and_eval_full_sup.sh 2>&1 | tee logs/mmca/rgb/mixup_train.log
