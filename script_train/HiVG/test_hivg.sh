#!/bin/bash

# ================= 全局参数 =================
IMGSIZE="224"
BATCHSIZE="32"
MODALITY="rgb"
CUDADEVICES="0,1"

export IMGSIZE
export BATCHSIZE
export MODALITY
export CUDADEVICES

echo "Start HIVG test with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

# ================= 日志目录 =================
mkdir -p logs/hivg_test/$MODALITY


echo "===== Start FLIR test ====="
DATASET="rgbtvg_flir"
export DATASET
stdbuf -oL -eL bash ./script_train/HiVG/test.sh  2>&1 | tee logs/hivg_test/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_flir.log"

echo "===== Start M3FD test ====="
DATASET="rgbtvg_m3fd"
export DATASET
stdbuf -oL -eL bash ./script_train/HiVG/test.sh  2>&1 | tee logs/hivg_test/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"


echo "===== Start MFAD test ====="
DATASET="rgbtvg_mfad"
export DATASET
stdbuf -oL -eL bash ./script_train/HiVG/test.sh  2>&1 | tee logs/hivg_test/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"


MODALITY="ir"
export MODALITY

echo "Start HIVG training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

# ================= 日志目录 =================
mkdir -p logs/hivg_test/$MODALITY


echo "===== Start FLIR test ====="
DATASET="rgbtvg_flir"
export DATASET
stdbuf -oL -eL bash ./script_train/HiVG/test.sh  2>&1 | tee logs/hivg_test/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_flir.log"

echo "===== Start M3FD test ====="
DATASET="rgbtvg_m3fd"
export DATASET
stdbuf -oL -eL bash ./script_train/HiVG/test.sh  2>&1 | tee logs/hivg_test/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"


echo "===== Start MFAD test ====="
DATASET="rgbtvg_mfad"
export DATASET
stdbuf -oL -eL bash ./script_train/HiVG/test.sh  2>&1 | tee logs/hivg_test/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"