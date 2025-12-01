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
  echo "Start OneRef_base_rec training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY"

  mkdir -p logs/OneRef_base_rec/$MODALITY

  # Select PRETRAIN_MODEL according to modality (currently same path for all modalities)
  export PRETRAIN_MODEL="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/b_rec_224.pth"

  for DATASET in "${DATASETS[@]}"; do
    export DATASET
    ds_name=${DATASET#rgbtvg_}
    echo "===== Start ${ds_name^^} training ====="
    stdbuf -oL -eL bash ./script_train/OneRef_base_rec/single.sh 2>&1 | tee logs/OneRef_base_rec/$MODALITY/${IMGSIZE}_${BATCHSIZE}_${ds_name}.log
  done
done
