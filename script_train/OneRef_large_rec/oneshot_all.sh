#!/bin/bash

IMGSIZE=${1:-224}
BATCHSIZE=${2:-32}
CUDADEVICES=${3:-0,1}
EPOCHS=${4:-110}

export IMGSIZE
export BATCHSIZE
export CUDADEVICES
export EPOCHS

DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir")

for MODALITY in "${MODALITIES[@]}"; do
  export MODALITY
  echo "Start OneRef_large_rec oneshot training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

  mkdir -p oneshot_logs/OneRef_large_rec/$MODALITY
  export PRETRAIN_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/l_rec_224.pth"

  for DATASET in "${DATASETS[@]}"; do
    export DATASET
    ds_name=${DATASET#rgbtvg_}
    echo "===== Start ${ds_name^^} oneshot training ====="
    stdbuf -oL -eL bash ./script_train/OneRef_large_rec/oneshot.sh 2>&1 | tee oneshot_logs/OneRef_large_rec/$MODALITY/${ds_name}.log
  done
done

