#!/bin/bash
# ===================== OneRef Visualization Script =====================
# Usage:
# bash visualize_scripts/shell_scripts/visualize_oneref.sh


# ===================== Configuration Parameters =====================
MODEL_CHECKPOINT="/home/xijiawen/code/rgbtvg/dataset_and_pretrain_model/pretrain_model/pretrained_weights/oneref/l_rec_224.pth"
MODEL="beit3_large_patch16_224"
TASK="grounding"
SENTENCEPIECE="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm"

# Data related
DATASET="rgbtvg_flir"
MODALITY="rgb"
LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_flir/rgbtvg_flir_val.pth"
DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/"

# Visualization parameters
OUTPUT_DIR="./visual_result/oneref_${DATASET}_${MODALITY}"
NUM_SAMPLES=0
START_IDX=0
IMSIZE=224

# GPU settings
GPU_ID="0"

# ===================== Run Visualization =====================
echo "Starting OneRef Visualization..."
echo "Dataset: $DATASET"
echo "Modality: $MODALITY"
echo "Model: $MODEL_CHECKPOINT"
echo "Label file: $LABEL_FILE"
echo "Data root: $DATAROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES (starting from $START_IDX)"
echo "----------------------------------------"

python visualize_scripts/oneref_visualize.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --model "$MODEL" \
    --task "$TASK" \
    --sentencepiece_model "$SENTENCEPIECE" \
    --label_file "$LABEL_FILE" \
    --dataroot "$DATAROOT" \
    --dataset "$DATASET" \
    --modality "$MODALITY" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --start_idx $START_IDX \
    --imsize $IMSIZE \
    --gpu_id "$GPU_ID" \
    --use_regress_box

echo "----------------------------------------"
echo "Visualization complete! Results saved to: $OUTPUT_DIR"
