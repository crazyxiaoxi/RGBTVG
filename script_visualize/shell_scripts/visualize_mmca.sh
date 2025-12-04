#!/bin/bash
# ===================== MMCA Visualization Script =====================
# Usage:
# bash visualize_scripts/shell_scripts/visualize_mmca.sh


# ===================== Parameter Parsing =====================
# Support command line arguments: DATASET / MODALITY / MODEL_CHECKPOINT
DATASET=${1:-"rgbtvg_flir"}
MODALITY=${2:-"rgb"}
MODEL_CHECKPOINT=${3:-"../dataset_and_pretrain_model/result/MMCA/MMCA_224_rgb_flir_best.pth"}

# ===================== Configuration Parameters =====================
BACKBONE="resnet50"
BERT_ENC_NUM=12
DETR_ENC_NUM=6

# Set LABEL_FILE and DATAROOT based on dataset and modality
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
OUTPUT_DIR="./visual_result/mmca/${DATASET}/${MODALITY}"
NUM_SAMPLES=0
START_IDX=0
IMSIZE=224

# GPU configuration
GPU_ID="0"

# ===================== Run Visualization =====================
echo "Starting MMCA Visualization..."
echo "Dataset: $DATASET"
echo "Modality: $MODALITY"
echo "Model: $MODEL_CHECKPOINT"
echo "Label file: $LABEL_FILE"
echo "Data root: $DATAROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES (starting from $START_IDX)"
echo "----------------------------------------"

python visualize_scripts/mmca_visualize.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --label_file "$LABEL_FILE" \
    --dataroot "$DATAROOT" \
    --dataset "$DATASET" \
    --modality "$MODALITY" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --start_idx $START_IDX \
    --imsize $IMSIZE \
    --gpu_id "$GPU_ID" \
    --backbone "$BACKBONE" \
    --bert_enc_num $BERT_ENC_NUM \
    --detr_enc_num $DETR_ENC_NUM \
    --max_query_len 20

echo "----------------------------------------"
echo "Visualization complete! Results saved to: $OUTPUT_DIR"
