# ================= Global hyper-parameters =================
IMGSIZE=${1:-224}
BATCHSIZE=${2:-32}
CUDADEVICES=${3:-0}

export IMGSIZE
export BATCHSIZE
export CUDADEVICES

DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir" "rgbt")

for MODALITY in "${MODALITIES[@]}"; do
  export MODALITY

  # Select RETRAIN checkpoint according to modality
  if [ "$MODALITY" == "rgbt" ]; then
      export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_large/mixup/best_checkpoint_rgbt.pth"
  elif [ "$MODALITY" == "rgb" ]; then
      export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_large/mixup/best_checkpoint.pth"
  elif [ "$MODALITY" == "ir" ]; then
      export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_large/mixup/best_checkpoint.pth"
  fi

  echo "Start HiVG-L training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY RETRAIN=$RETRAIN"

  # Log directory
  mkdir -p logs/hivg_l/$MODALITY

  for DATASET in "${DATASETS[@]}"; do
    ds_name=${DATASET#rgbtvg_}
    echo "===== Start ${ds_name^^} training ====="

    if [ "$DATASET" == "rgbtvg_flir" ]; then
      stdbuf -oL -eL bash ./script_train/HiVG_L/rgbtvg_flir/retrain_hivg_single_dataset_flir_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg_l/$MODALITY/${IMGSIZE}_${BATCHSIZE}_flir.log
    elif [ "$DATASET" == "rgbtvg_m3fd" ]; then
      stdbuf -oL -eL bash ./script_train/HiVG_L/rgbtvg_m3fd/retrain_hivg_single_dataset_m3fd_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg_l/$MODALITY/${IMGSIZE}_${BATCHSIZE}_m3fd.log
    elif [ "$DATASET" == "rgbtvg_mfad" ]; then
      stdbuf -oL -eL bash ./script_train/HiVG_L/rgbtvg_mfad/retrain_hivg_single_dataset_mfad_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg_l/$MODALITY/${IMGSIZE}_${BATCHSIZE}_mfad.log
    fi
  done
done
