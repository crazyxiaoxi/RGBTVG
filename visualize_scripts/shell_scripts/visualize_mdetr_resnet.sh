#!/bin/bash
# ===================== MDETR-ResNet可视化脚本 =====================
# 使用示例：
# bash visualize_scripts/shell_scripts/visualize_mdetr_resnet.sh

# 激活conda环境（如果需要）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rgbtvg

# ===================== 配置参数 =====================
# 模型相关
# 注意：请根据你的实际模型路径修改以下路径
MODEL_CHECKPOINT="/home/xijiawen/code/rgbtvg/dataset_and_pretrain_model/result/MDETR_res/MDETR_resnet_224_ir_flir_best.pth"  # 修改为你的模型路径
MODEL_TYPE="ResNet"  # ResNet 或 CLIP
BACKBONE="resnet50"  # resnet50 或其他
BERT_ENC_NUM=12
DETR_ENC_NUM=6

# 数据集相关
DATASET="rgbtvg_flir"  # 数据集名称
MODALITY="ir"  # rgb, ir, rgbt - 根据你想用的图像模态选择
LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_flir/rgbtvg_flir_train.pth"
DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/ir/"  # 图像目录

# 可视化参数
OUTPUT_DIR="./visual_result/mdetr_resnet_${DATASET}_${MODALITY}"
NUM_SAMPLES=100  # 可视化样本数
START_IDX=0      # 起始索引
IMSIZE=224       # 图像尺寸（将使用checkpoint中的配置）

# GPU配置
GPU_ID="0"

# ===================== 运行可视化 =====================
echo "Starting MDETR-ResNet Visualization..."
echo "Dataset: $DATASET"
echo "Modality: $MODALITY"
echo "Model Type: $MODEL_TYPE"
echo "Model: $MODEL_CHECKPOINT"
echo "Label file: $LABEL_FILE"
echo "Data root: $DATAROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Samples: $NUM_SAMPLES (starting from $START_IDX)"
echo "----------------------------------------"

python visualize_scripts/mdetr_visualize.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --label_file "$LABEL_FILE" \
    --dataroot "$DATAROOT" \
    --dataset "$DATASET" \
    --modality "$MODALITY" \
    --model_type "$MODEL_TYPE" \
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
