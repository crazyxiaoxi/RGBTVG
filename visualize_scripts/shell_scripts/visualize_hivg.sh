#!/bin/bash
# ===================== HiVG可视化脚本 =====================
# 使用示例：
# bash visualize_scripts/shell_scripts/visualize_hivg.sh

# 激活conda环境（如果需要）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rgbtvg

# ===================== 配置参数 =====================
# 模型相关
MODEL_CHECKPOINT="/home/xijiawen/code/rgbtvg/dataset_and_pretrain_model/pretrain_model/pretrained_weights/HiVG/mixup_pretraining_base/best_checkpoint.pth"  # 修改为你的模型路径
MODEL="ViT-B/16"  # ViT-B/16 或 ViT-L/14
TASK="grounding"

# 数据相关
DATASET="rgbtvg_flir"  # 数据集名称: rgbtvg_flir, rgbtvg_m3fd, rgbtvg_mfad
MODALITY="rgb"         # 模态: rgb, ir, rgbt
LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_flir/rgbtvg_flir_train.pth"
DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/"

# 可视化参数
OUTPUT_DIR="./visual_result/hivg_${DATASET}_${MODALITY}"
NUM_SAMPLES=100        # 要可视化的样本数量（0表示全部）
START_IDX=0            # 起始索引
IMSIZE=224             # 图像大小

# GPU设置
GPU_ID="0"             # 使用的GPU ID

# ===================== 运行可视化 =====================
echo "Starting HiVG Visualization..."
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
    --enable_adaptive_weights \
    --use_mask_loss

echo "----------------------------------------"
echo "Visualization complete! Results saved to: $OUTPUT_DIR"
