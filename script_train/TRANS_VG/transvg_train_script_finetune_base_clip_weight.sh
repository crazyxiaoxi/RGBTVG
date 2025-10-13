#!/bin/bash

export IMGSIZE=224
export BATCHSIZE=16
MODALITY='rgb'
export CUDADEVICES=1,2,3

mkdir -p logs/transvg/rgb

echo "===== Start TransVG FLIR training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/rgbt_flir/transvg_single_dataset_flir_train_and_eval_full_sup.sh 2>&1 | tee logs/transvg/rgb/flir_train.log

echo "===== Start TransVG M3FD training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/rgbt_m3fd/transvg_single_dataset_m3fd_train_and_eval_full_sup.sh 2>&1 | tee logs/transvg/rgb/m3fd_train.log

echo "===== Start TransVG MFAD training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/rgbt_mfad/transvg_single_dataset_mfad_train_and_eval_full_sup.sh 2>&1 | tee logs/transvg/rgb/mfad_train.log

echo "===== Start TransVG MIXUP training ====="
stdbuf -oL -eL bash ./script_train/TRANS_VG/rgbtvg_mixup/transvg_mixup_dataset_train_and_eval_full_sup.sh 2>&1 | tee logs/transvg/rgb/mixup_train.log
