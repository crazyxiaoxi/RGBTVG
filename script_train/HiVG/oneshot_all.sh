
# ================= 全局参数 =================
IMGSIZE=${1:-224}
BATCHSIZE=${2:-36}
MODALITY=${3:-rgb}
CUDADEVICES=${4:-0}

export IMGSIZE
export BATCHSIZE
export MODALITY
export CUDADEVICES

if [ "$MODALITY" == "rgbt" ]; then
    export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/best_checkpoint_rgbt.pth"
elif [ "$MODALITY" == "rgb" ]; then
    export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/best_checkpoint.pth"
elif [ "$MODALITY" == "ir" ]; then
    export RETRAIN="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/best_checkpoint.pth"
fi


echo "Start HIVG training with IMGSIZE=$IMGSIZE BATCHSIZE=$BATCHSIZE CUDA=$CUDADEVICES MODALITY=$MODALITY RETRAIN=$RETRAIN"

# ================= 日志目录 =================
mkdir -p oneshot_logs/hivg/$MODALITY

# ================= 串行训练四个数据集 =================


echo "===== Start FLIR training ====="
DATASET="rgbtvg_flir"
export DATASET
stdbuf -oL -eL bash ./script_train/HiVG/oneshot.sh  2>&1 | tee oneshot_logs/hivg/$MODALITY/flir.log

echo "===== Start M3FD training ====="
DATASET="rgbtvg_m3fd"
export DATASET
stdbuf -oL -eL bash ./script_train/HiVG/oneshot.sh 2>&1 | tee oneshot_logs/hivg/$MODALITY/m3fd.log

echo "===== Start MFAD training ====="
DATASET="rgbtvg_mfad"
export DATASET
stdbuf -oL -eL bash ./script_train/HiVG/oneshot.sh 2>&1 | tee oneshot_logs/hivg/$MODALITY/mfad.log

