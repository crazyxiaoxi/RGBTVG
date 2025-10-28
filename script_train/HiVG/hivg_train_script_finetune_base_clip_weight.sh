#!/bin/bash

# ================= 全局参数 =================
IMGSIZE=${1:-224}
BATCHSIZE=${2:-32}
MODALITY=${3:-rgbt}
CUDADEVICES=${4:-0}

export IMGSIZE
export BATCHSIZE
export MODALITY
export CUDADEVICES

echo "Start HIVG training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

# ================= 日志目录 =================
mkdir -p logs/hivg/$MODALITY

# ================= 串行训练四个数据集 =================

echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/HiVG/rgbtvg_flir/hivg_single_dataset_flir_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_flir.log"

echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/HiVG/rgbtvg_m3fd/hivg_single_dataset_m3fd_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"

echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/HiVG/rgbtvg_mfad/hivg_single_dataset_mfad_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mfad.log"

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/HiVG/rgbtvg_mixup/hivg_mixup_dataset_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mixup.log"
