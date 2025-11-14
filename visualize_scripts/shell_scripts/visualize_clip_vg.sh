#!/bin/bash
# ===================== CLIP_VG可视化脚本 =====================
# 使用示例：
# bash visualize_scripts/shell_scripts/visualize_clip_vg.sh [DATASET] [MODALITY] [MODEL_CHECKPOINT]
# 或者直接运行使用默认参数

# 激活conda环境（如果需要）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rgbtvg

# ===================== 参数解析 =====================
# 从命令行参数获取，如果没有则使用默认值
DATASET=${1:-"rgbtvg_flir"}  # 默认rgbtvg_flir
MODALITY=${2:-"rgb"}         # 默认rgb
MODEL_CHECKPOINT=${3:-"/home/xijiawen/code/rgbtvg/dataset_and_pretrain_model/result/clip_vg/CLIP_VG_224_rgb_flir_best.pth"}

# ===================== 配置参数 =====================
# 模型相关
MODEL_TYPE="ViT-B/16"  # ViT-B/16 或 ViT-L/14
VL_HIDDEN_DIM=512
VL_ENC_LAYERS=6

# 根据数据集设置数据路径
case $DATASET in
    "rgbtvg_flir")
        LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_flir/rgbtvg_flir_train.pth"
        case $MODALITY in
            "rgb") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/" ;;
            "ir") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/ir/" ;;
            "rgbt") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/" ;;
        esac
        ;;
    "rgbtvg_m3fd")
        LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_m3fd/rgbtvg_m3fd_train.pth"
        case $MODALITY in
            "rgb") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb/" ;;
            "ir") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/ir/" ;;
            "rgbt") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb/" ;;
        esac
        ;;
    "rgbtvg_mfad")
        LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_mfad/rgbtvg_mfad_train.pth"
        case $MODALITY in
            "rgb") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/rgb/" ;;
            "ir") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/ir/" ;;
            "rgbt") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/rgb/" ;;
        esac
        ;;
esac

# 可视化参数 - 新的目录结构：模型名称/数据集/模态
OUTPUT_DIR="./visual_result/clip_vg/${DATASET}/${MODALITY}"
NUM_SAMPLES=300  # 可视化样本数
START_IDX=0      # 起始索引
IMSIZE=224       # 图像尺寸

# GPU配置
GPU_ID="0"

# ===================== 运行可视化 =====================
echo "Starting CLIP_VG Visualization..."
echo "Dataset: $DATASET"
echo "Modality: $MODALITY"
echo "Model Type: $MODEL_TYPE"
echo "Model: $MODEL_CHECKPOINT"
echo "Label file: $LABEL_FILE"
echo "Data root: $DATAROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES (starting from $START_IDX)"
echo "----------------------------------------"

python visualize_scripts/clip_vg_visualize.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --label_file "$LABEL_FILE" \
    --dataroot "$DATAROOT" \
    --dataset "$DATASET" \
    --modality "$MODALITY" \
    --model "$MODEL_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES \
    --start_idx $START_IDX \
    --imsize $IMSIZE \
    --gpu_id "$GPU_ID" \
    --vl_hidden_dim $VL_HIDDEN_DIM \
    --vl_enc_layers $VL_ENC_LAYERS \
    --max_query_len 77 \
    --prompt '{pseudo_query}'

echo "----------------------------------------"
echo "Visualization complete! Results saved to: $OUTPUT_DIR"
