#!/bin/bash

export IMGSIZE=224
export BATCHSIZE=16
MODALITY='rgb'
export CUDADEVICES=1,2,3

mkdir -p logs/clipvg/rgb

echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/rgbt_flir/clipvg_single_dataset_flir_train_and_eval_full_sup.sh 2>&1 | tee logs/clipvg/rgb/flir_train.log

echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/rgbt_m3fd/clipvg_single_dataset_m3fd_train_and_eval_full_sup.sh 2>&1 | tee logs/clipvg/rgb/m3fd_train.log

echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/rgbt_mfad/clipvg_single_dataset_mfad_train_and_eval_full_sup.sh 2>&1 | tee logs/clipvg/rgb/mfad_train.log

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/CLIP_VG/rgbtvg_mixup/clipvg_mixup_dataset_train_and_eval_full_sup.sh 2>&1 | tee logs/clipvg/rgb/mixup_train.log
