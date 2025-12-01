#!/bin/bash

IMGSIZE=${1:-224}
BATCHSIZE=${2:-8}
CUDADEVICES=${3:-1,2,3}
EPOCHS=${4:-110}


# Export common hyper-parameters for the underlying single.sh script
export IMGSIZE
export BATCHSIZE
export CUDADEVICES
export EPOCHS

DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir" "rgbt")

for MODALITY in "${MODALITIES[@]}"; do
  export MODALITY
  echo "Start CLIPVG training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

  mkdir -p logs/clipvg/$MODALITY

  for DATASET in "${DATASETS[@]}"; do
    export DATASET
    ds_name=${DATASET#rgbtvg_}
    echo "===== Start ${ds_name^^} training ====="
    stdbuf -oL -eL bash ./script_train/CLIP_VG/single.sh 2>&1 | tee logs/clipvg/$MODALITY/${IMGSIZE}_${BATCHSIZE}_${ds_name}.log
  done
done
