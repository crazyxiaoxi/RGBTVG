#!/bin/bash
# ===================== OneRef Base 可视化脚本 =====================
# 使用示例：
# bash visualize_scripts/shell_scripts/visualize_oneref_base.sh [DATASET] [MODALITY] [MODEL_CHECKPOINT]
# DATASET: rgbtvg_flir, rgbtvg_m3fd, rgbtvg_mfad
# MODALITY: rgb, ir, rgbt

# ===================== 参数解析 =====================
DATASET=${1:-"rgbtvg_flir"}
MODALITY=${2:-"rgb"}
MODEL_CHECKPOINT=${3:-"../dataset_and_pretrain_model/result/OneRef_B/OneRef_B_rgb_flir_best.pth"}

# 模型相关（Base 尺度 224）
MODEL="beit3_base_patch16_224"
TASK="grounding"
SENTENCEPIECE="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm"

# 根据数据集和模态设置 LABEL_FILE 与 DATAROOT
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
        echo "❌ 错误: 不支持的数据集 $DATASET"
        echo "支持的数据集: rgbtvg_flir, rgbtvg_m3fd, rgbtvg_mfad"
        exit 1
        ;;
 esac

OUTPUT_DIR="./visual_result/oneref_base/${DATASET}/${MODALITY}"
NUM_SAMPLES=0
START_IDX=0
IMSIZE=224
GPU_ID="0"

# ===================== 运行可视化 =====================
echo "Starting OneRef Base Visualization..."
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
    --max_query_len 64 \
    --gpu_id "$GPU_ID" \
    --use_regress_box

echo "----------------------------------------"
echo "OneRef Base Visualization complete! Results saved to: $OUTPUT_DIR"
