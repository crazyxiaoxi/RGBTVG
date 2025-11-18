#!/bin/bash
# ===================== CLIP_VG全面测试脚本 =====================
# 自动测试所有数据集和模态组合
# 使用示例：bash run_all_clip_vg_tests.sh

echo "🚀 开始CLIP_VG全面测试..."
echo "测试范围："
echo "  - 数据集: flir, m3fd, mfad"
echo "  - 模态: rgb, ir, rgbt"
echo "  - 总计: 9种组合"
echo "========================================"

# 定义数据集和模态
DATASETS=("rgbtvg_flir" "rgbtvg_m3fd" "rgbtvg_mfad")
MODALITIES=("rgb" "ir" "rgbt")
# 模型路径基础目录
MODEL_BASE_PATH="../dataset_and_pretrain_model/result/clip_vg"

# 计数器
TOTAL_TESTS=9
CURRENT_TEST=0
SUCCESS_COUNT=0
FAILED_TESTS=()

# 开始时间
START_TIME=$(date +%s)

# 遍历所有组合
for dataset in "${DATASETS[@]}"; do
    for modality in "${MODALITIES[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        
        # 构建模型路径
        # 从数据集名称提取简短名称 (rgbtvg_flir -> flir)
        DATASET_SHORT=$(echo $dataset | sed 's/rgbtvg_//')
        MODEL_CHECKPOINT="${MODEL_BASE_PATH}/CLIP_VG_224_${modality}_${DATASET_SHORT}_best.pth"
        
        echo ""
        echo "📊 测试 $CURRENT_TEST/$TOTAL_TESTS: $dataset + $modality"
        echo "   模型: $MODEL_CHECKPOINT"
        echo "   输出: ./visual_result/clip_vg/$dataset/$modality"
        echo "----------------------------------------"
        
        # 检查模型文件是否存在
        if [ ! -f "$MODEL_CHECKPOINT" ]; then
            echo "❌ 警告: 模型文件不存在: $MODEL_CHECKPOINT"
            echo "   跳过此测试..."
            FAILED_TESTS+=("$dataset-$modality (模型文件不存在)")
            continue
        fi
        
        # 运行测试
        if bash visualize_scripts/shell_scripts/visualize_clip_vg.sh "$dataset" "$modality" "$MODEL_CHECKPOINT"; then
            echo "✅ 测试成功: $dataset + $modality"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "❌ 测试失败: $dataset + $modality"
            FAILED_TESTS+=("$dataset-$modality")
        fi
        
        echo "----------------------------------------"
        
        # 短暂暂停避免GPU过载
        sleep 2
    done
done

# 结束时间和统计
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "🎉 CLIP_VG全面测试完成！"
echo "========================================"
echo "📈 测试统计:"
echo "   总测试数: $TOTAL_TESTS"
echo "   成功: $SUCCESS_COUNT"
echo "   失败: $((TOTAL_TESTS - SUCCESS_COUNT))"
echo "   耗时: ${MINUTES}分${SECONDS}秒"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "❌ 失败的测试:"
    for failed in "${FAILED_TESTS[@]}"; do
        echo "   - $failed"
    done
fi

echo ""
echo "📁 结果目录结构:"
echo "visual_result/clip_vg/"
echo "├── rgbtvg_flir/"
echo "│   ├── rgb/     # RGB模态结果"
echo "│   ├── ir/      # IR模态结果"
echo "│   └── rgbt/    # RGBT模态结果 (双图像输出)"
echo "├── rgbtvg_m3fd/"
echo "│   ├── rgb/"
echo "│   ├── ir/"
echo "│   └── rgbt/"
echo "└── rgbtvg_mfad/"
echo "    ├── rgb/"
echo "    ├── ir/"
echo "    └── rgbt/"

echo ""
echo "💡 提示:"
echo "   - RGBT模态会生成两张图片: *_rgb.jpg 和 *_ir.jpg"
echo "   - 每个测试默认生成100个样本"
echo "   - 如需修改样本数，请编辑 visualize_clip_vg.sh 中的 NUM_SAMPLES"

if [ $SUCCESS_COUNT -eq $TOTAL_TESTS ]; then
    echo ""
    echo "🎊 所有测试都成功完成！"
    exit 0
else
    echo ""
    echo "⚠️  部分测试失败，请检查上述失败列表"
    exit 1
fi
