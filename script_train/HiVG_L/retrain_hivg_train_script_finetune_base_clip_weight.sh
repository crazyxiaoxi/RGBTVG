
# ================= 全局参数 =================
IMGSIZE=${1:-224}
BATCHSIZE=${2:-32}
MODALITY=${3:-rgbt}
CUDADEVICES=${4:-0}

export IMGSIZE
export BATCHSIZE
export MODALITY
export CUDADEVICES

if [ "$MODALITY" == "rgbt" ]; then
    export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_large/mixup/best_checkpoint_rgbt.pth"
elif [ "$MODALITY" == "rgb" ]; then
    export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_large/mixup/best_checkpoint.pth"
elif [ "$MODALITY" == "ir" ]; then
    export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_large/mixup/best_checkpoint.pth"
fi


echo "Start HIVG training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY RETRAIN=$RETRAIN"

# ================= 日志目录 =================
mkdir -p logs/hivg_l/$MODALITY

# ================= 串行训练四个数据集 =================

echo "===== Start FLIR training ====="
stdbuf -oL -eL bash ./script_train/HiVG_L/rgbtvg_flir/retrain_hivg_single_dataset_flir_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg_l/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_flir.log"

echo "===== Start M3FD training ====="
stdbuf -oL -eL bash ./script_train/HiVG_L/rgbtvg_m3fd/retrain_hivg_single_dataset_m3fd_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg_l/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_m3fd.log"

echo "===== Start MFAD training ====="
stdbuf -oL -eL bash ./script_train/HiVG_L/rgbtvg_mfad/retrain_hivg_single_dataset_mfad_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg_l/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mfad.log"

echo "===== Start MIXUP training ====="
stdbuf -oL -eL bash ./script_train/HiVG_L/rgbtvg_mixup/retrain_hivg_mixup_dataset_finetune_base_clip_weight.sh $IMGSIZE $BATCHSIZE $MODALITY $CUDADEVICES 2>&1 | tee logs/hivg_l/$MODALITY/$IMGSIZE"_"$BATCHSIZE"_mixup.log"
