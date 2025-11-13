#!/bin/bash
# ===================== TransVG可视化脚本 =====================
# 使用示例：
# bash visualize_scripts/shell_scripts/visualize_transvg.sh [DATASET] [MODALITY] [MODEL_CHECKPOINT]
# 或者直接运行使用默认参数

# 激活conda环境（如果需要）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rgbtvg

# ===================== 参数解析 =====================
# 从命令行参数获取，如果没有则使用默认值
DATASET=${1:-"rgbtvg_flir"}  # 默认rgbtvg_flir
MODALITY=${2:-"rgb"}          # 默认rgb
MODEL_CHECKPOINT=${3:-"/home/xijiawen/code/rgbtvg/dataset_and_pretrain_model/result/transvg/TransVG_224_rgb_flir_best.pth"}

# ===================== 配置参数 =====================
# 模型相关
BACKBONE="resnet50"  # resnet50 或其他
BERT_ENC_NUM=12
DETR_ENC_NUM=6

# 根据数据集设置数据路径
case $DATASET in
    "rgbtvg_flir")
        LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_flir/rgbtvg_flir_train.pth"
        case $MODALITY in
            "rgb") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/" ;;
            "ir") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/ir/" ;;
            "rgbt") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/" ;;  # RGBT使用RGB目录
        esac
        ;;
    "rgbtvg_m3fd")
        LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_m3fd/rgbtvg_m3fd_train.pth"
        case $MODALITY in
            "rgb") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb/" ;;
            "ir") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/ir/" ;;
            "rgbt") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb/" ;;  # RGBT使用RGB目录
        esac
        ;;
    "rgbtvg_mfad")
        LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_mfad/rgbtvg_mfad_train.pth"
        case $MODALITY in
            "rgb") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/rgb/" ;;
            "ir") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/ir/" ;;
            "rgbt") DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/rgb/" ;;  # RGBT使用RGB目录
        esac
        ;;
    *)
        echo "❌ 错误: 不支持的数据集 $DATASET"
        echo "支持的数据集: rgbtvg_flir, rgbtvg_m3fd, rgbtvg_mfad"
        exit 1
        ;;
esac

# 可视化参数
OUTPUT_DIR="./visual_result/transvg/${DATASET}/${MODALITY}"
NUM_SAMPLES=100  # 可视化样本数
START_IDX=0      # 起始索引
IMSIZE=224       # 图像尺寸 (根据模型名称TransVG_224调整)

# GPU配置
GPU_ID="0"

# ===================== 运行可视化 =====================
echo "Starting TransVG Visualization..."
echo "Dataset: $DATASET"
echo "Modality: $MODALITY"
echo "Model: $MODEL_CHECKPOINT"
echo "Label file: $LABEL_FILE"
echo "Data root: $DATAROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES (starting from $START_IDX)"
echo "----------------------------------------"

python visualize_scripts/transvg_visualize.py \
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
