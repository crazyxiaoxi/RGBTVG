#!/bin/bash
# ===================== HiVG Base Visualization Script =====================
# Usage:
# bash visualize_scripts/shell_scripts/visualize_hivg_base.sh [DATASET] [MODALITY] [MODEL_CHECKPOINT]
# DATASET: rgbtvg_flir, rgbtvg_m3fd, rgbtvg_mfad
# MODALITY: rgb, ir, rgbt

# ===================== Parameter Parsing =====================
DATASET=${1:-"rgbtvg_flir"}
MODALITY=${2:-"rgb"}
MODEL_CHECKPOINT=${3:-"../dataset_and_pretrain_model/result/HiVG_B/HiVG_B_rgb_flir_best.pth"}

# ===================== Configuration Parameters =====================
# Model related (Base: ViT-B/16)
MODEL="ViT-B/16"

# Set data paths based on dataset
case $DATASET in
    "rgbtvg_flir")
        LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_flir/rgbtvg_flir_val.pth"
        case $MODALITY in
            "rgb")  DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/" ;;
            "ir")   DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/ir/" ;;
            "rgbt") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/" ;;
        esac
        ;;
    "rgbtvg_m3fd")
        LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_m3fd/rgbtvg_m3fd_val.pth"
        case $MODALITY in
            "rgb")  DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb/" ;;
            "ir")   DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/ir/" ;;
            "rgbt") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb/" ;;
        esac
        ;;
    "rgbtvg_mfad")
        LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_mfad/rgbtvg_mfad_val.pth"
        case $MODALITY in
            "rgb")  DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/rgb/" ;;
            "ir")   DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/ir/" ;;
            "rgbt") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/rgb/" ;;
        esac
        ;;
    *)
        echo "Error: Unsupported dataset $DATASET"
        echo "Supported datasets: rgbtvg_flir, rgbtvg_m3fd, rgbtvg_mfad"
        exit 1
        ;;
 esac

# Visualization parameters
OUTPUT_DIR="./visual_result/hivg_base/${DATASET}/${MODALITY}"
NUM_SAMPLES=0
START_IDX=0
IMSIZE=224
GPU_ID="0"

# ===================== Run Visualization =====================
echo "Starting HiVG Base Visualization..."
echo "Dataset: $DATASET"
echo "Modality: $MODALITY"
echo "Model: $MODEL_CHECKPOINT"
echo "Label file: $LABEL_FILE"
echo "Data root: $DATAROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES (starting from $START_IDX)"
echo "----------------------------------------"

python visualize_scripts/hivg_visualize.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --model "$MODEL" \
    --label_file "$LABEL_FILE" \
    --dataroot "$DATAROOT" \
    --dataset "$DATASET" \
    --modality "$MODALITY" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --start_idx $START_IDX \
    --imsize $IMSIZE \
    --gpu_id "$GPU_ID" \
    --vl_hidden_dim 512 \
    --normalize_before \
    --hi_lora_stage 3 \
    --enable_adaptive_weights \
    --max_query_len 77 \
    --use_mask_loss

echo "----------------------------------------"
echo "HiVG Base Visualization complete! Results saved to: $OUTPUT_DIR"
