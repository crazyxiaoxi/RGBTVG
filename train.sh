#!/bin/bash
# ============================================================
# 总启动脚本：依次运行所有模型的训练与测试脚本
# ============================================================
echo -e "\n\n===================== 启动全部训练与测试 ====================="

echo -e "\n\n===== 启动 CLIPVG 训练与测试 ====="
bash /workspace/xijiawen/code/rgbtvg/RGBTVG-Benchmark/script_train/CLIP_VG/clipvg_train_script_finetune_base_clip_weight.sh

echo -e "\n\n===== 启动 HIVG 训练与测试 ====="
bash /workspace/xijiawen/code/rgbtvg/RGBTVG-Benchmark/script_train/HiVG/hivg_train_script_finetune_base_clip_weight.sh

echo -e "\n\n===== 启动 MDETR 训练与测试 ====="
bash /workspace/xijiawen/code/rgbtvg/RGBTVG-Benchmark/script_train/MDETR/mdetr_train_script_finetune_base_clip_weight.sh

echo -e "\n\n===== 启动 MMCA 训练与测试 ====="
bash /workspace/xijiawen/code/rgbtvg/RGBTVG-Benchmark/script_train/MMCA_VG/mmca_train_script_finetune_base_clip_weight.sh

echo -e "\n\n===== 启动 TRANSVG 训练与测试 ====="
bash /workspace/xijiawen/code/rgbtvg/RGBTVG-Benchmark/script_train/TRANS_VG/transvg_train_script_finetune_base_clip_weight.sh
# ---- 结束 ----
echo -e "\n\n===================== 所有任务已执行完毕 ====================="
