#!/bin/bash

export IMGSIZE=224
export BATCHSIZE=4
MODALITY='rgbt'
export CUDADEVICES=3

mkdir -p logs/transvg/rgbt

echo "===== Start TransVG FLIR training ====="
stdbuf -oL -eL bash ./script_train/TransVG/rgbt_flir/clipvg_single_dataset_flir_train_and_eval_transvg.sh 2>&1 | tee logs/transvg/rgbt/flir_train.log

echo "===== Start TransVG M3FD training ====="
stdbuf -oL -eL bash ./script_train/TransVG/rgbt_m3fd/clipvg_single_dataset_m3fd_train_and_eval_transvg.sh 2>&1 | tee logs/transvg/rgbt/m3fd_train.log

echo "===== Start TransVG MFAD training ====="
stdbuf -oL -eL bash ./script_train/TransVG/rgbt_mfad/clipvg_single_dataset_mfad_train_and_eval_transvg.sh 2>&1 | tee logs/transvg/rgbt/mfad_train.log

echo "===== Start TransVG MIXUP training ====="
stdbuf -oL -eL bash ./script_train/TransVG/rgbtvg_mixup/clipvg_mixup_dataset_train_and_eval_transvg.sh 2>&1 | tee logs/transvg/rgbt/mixup_train.log
