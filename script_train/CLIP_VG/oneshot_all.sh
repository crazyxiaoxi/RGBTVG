#!/bin/bash

IMGSIZE=${1:-224}
BATCHSIZE=${2:-8}
CUDADEVICES=${3:-0}
EPOCHS=${4:-110}


# Export common hyper-parameters for the underlying oneshot.sh script
export IMGSIZE
export BATCHSIZE
export CUDADEVICES
export EPOCHS

DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir")

for MODALITY in "${MODALITIES[@]}"; do
  export MODALITY
  echo "Start CLIPVG oneshot training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

  mkdir -p oneshot_logs/clipvg/$MODALITY

  for DATASET in "${DATASETS[@]}"; do
    export DATASET
    ds_name=${DATASET#rgbtvg_}
    echo "===== Start ${ds_name^^} oneshot training ====="
    stdbuf -oL -eL bash ./script_train/CLIP_VG/oneshot.sh 2>&1 | tee oneshot_logs/clipvg/$MODALITY/${ds_name}.log
  done
done
