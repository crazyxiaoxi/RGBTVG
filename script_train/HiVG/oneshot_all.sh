# ================= Global hyper-parameters =================
IMGSIZE=${1:-224}
BATCHSIZE=${2:-36}
CUDADEVICES=${3:-0}

export IMGSIZE
export BATCHSIZE
export CUDADEVICES

DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir")

for MODALITY in "${MODALITIES[@]}"; do
  export MODALITY

  # Select RETRAIN checkpoint according to modality
  if [ "$MODALITY" == "rgbt" ]; then
      export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/best_checkpoint_rgbt.pth"
  elif [ "$MODALITY" == "rgb" ]; then
      export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/best_checkpoint.pth"
  elif [ "$MODALITY" == "ir" ]; then
      export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/best_checkpoint.pth"
  fi

  echo "Start HiVG-B oneshot training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY RETRAIN=$RETRAIN"

  # Log directory
  mkdir -p oneshot_logs/hivg/$MODALITY

  for DATASET in "${DATASETS[@]}"; do
    export DATASET
    ds_name=${DATASET#rgbtvg_}
    echo "===== Start ${ds_name^^} oneshot training ====="
    stdbuf -oL -eL bash ./script_train/HiVG/oneshot.sh  2>&1 | tee oneshot_logs/hivg/$MODALITY/${ds_name}.log
  done
done

