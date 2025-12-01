#!/bin/bash

IMGSIZE=${1:-224}
BATCHSIZE=${2:-32}
CUDADEVICES=${3:-0}
EPOCHS=${4:-110}

export IMGSIZE
export BATCHSIZE
export CUDADEVICES
export EPOCHS

DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir" "rgbt")

for MODALITY in "${MODALITIES[@]}"; do
  export MODALITY
  echo "Start MDETR CLIP training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

  mkdir -p logs/mdetr_clip/$MODALITY

  for DATASET in "${DATASETS[@]}"; do
    export DATASET
    ds_name=${DATASET#rgbtvg_}
    echo "===== Start ${ds_name^^} training ====="
    stdbuf -oL -eL bash ./script_train/MDETR_clip/single.sh 2>&1 | tee logs/mdetr_clip/$MODALITY/${IMGSIZE}_${BATCHSIZE}_${ds_name}.log
  done
done
