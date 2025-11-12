# MDETR_resnet批量测试指南

## 📋 概述

MDETR_resnet批量测试功能可以自动测试MDETR_resnet模型在所有数据集和模态组合下的可视化效果。

## 🎯 测试范围

### 数据集 (3个)
- `rgbtvg_flir` - FLIR数据集
- `rgbtvg_m3fd` - M3FD数据集  
- `rgbtvg_mfad` - MFAD数据集

### 模态 (3个)
- `rgb` - RGB图像
- `ir` - 红外图像
- `rgbt` - RGB+红外融合图像

### 总计: **9种组合**

## 🚀 使用方法

### 1. 运行全部测试
```bash
# 在RGBTVG根目录下运行
bash run_all_mdetr_resnet_tests.sh
```

### 2. 运行单个测试
```bash
# 格式: bash visualize_scripts/shell_scripts/visualize_mdetr_resnet.sh [DATASET] [MODALITY] [MODEL_PATH]
bash visualize_scripts/shell_scripts/visualize_mdetr_resnet.sh rgbtvg_flir ir
bash visualize_scripts/shell_scripts/visualize_mdetr_resnet.sh rgbtvg_m3fd rgbt
```

## 📁 目录结构

### 输出目录结构
```
visual_result/mdetr_resnet/
├── rgbtvg_flir/
│   ├── rgb/                    # FLIR RGB模态结果
│   │   ├── mdetr_pred_000000.jpg
│   │   ├── mdetr_pred_000000.txt
│   │   └── ...
│   ├── ir/                     # FLIR IR模态结果
│   │   ├── mdetr_pred_000000.jpg
│   │   ├── mdetr_pred_000000.txt
│   │   └── ...
│   └── rgbt/                   # FLIR RGBT模态结果 (双图像)
│       ├── mdetr_pred_000000_rgb.jpg  # RGB彩色图
│       ├── mdetr_pred_000000_ir.jpg   # IR灰度图
│       ├── mdetr_pred_000000.txt      # 文本描述
│       └── ...
├── rgbtvg_m3fd/
│   ├── rgb/
│   ├── ir/
│   └── rgbt/
└── rgbtvg_mfad/
    ├── rgb/
    ├── ir/
    └── rgbt/
```

### 模型文件路径
```
dataset_and_pretrain_model/result/MDETR_resnet/
├── MDETR_resnet_224_rgb_flir_best.pth      # FLIR RGB模型
├── MDETR_resnet_224_ir_flir_best.pth       # FLIR IR模型
├── MDETR_resnet_224_rgbt_flir_best.pth     # FLIR RGBT模型
├── MDETR_resnet_224_rgb_m3fd_best.pth      # M3FD RGB模型
├── MDETR_resnet_224_ir_m3fd_best.pth       # M3FD IR模型
├── MDETR_resnet_224_rgbt_m3fd_best.pth     # M3FD RGBT模型
├── MDETR_resnet_224_rgb_mfad_best.pth      # MFAD RGB模型
├── MDETR_resnet_224_ir_mfad_best.pth       # MFAD IR模型
└── MDETR_resnet_224_rgbt_mfad_best.pth     # MFAD RGBT模型
```

## ⚙️ 配置参数

### 默认配置
- **样本数**: 100个样本
- **起始索引**: 0
- **图像尺寸**: 224x224
- **GPU**: GPU 0
- **模型类型**: ResNet
- **骨干网络**: resnet50

### 修改配置
编辑 `visualize_scripts/shell_scripts/visualize_mdetr_resnet.sh`:
```bash
NUM_SAMPLES=100     # 修改样本数
START_IDX=0         # 修改起始索引
GPU_ID="0"          # 修改GPU ID
BACKBONE="resnet50" # 修改骨干网络
```

## 🔧 数据路径映射

脚本会自动根据数据集和模态设置正确的数据路径：

### FLIR数据集
- **Label**: `../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_flir/rgbtvg_flir_train.pth`
- **RGB**: `../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/`
- **IR**: `../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/ir/`
- **RGBT**: `../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/flir/rgb/` (用于配对)

### M3FD数据集
- **Label**: `../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_m3fd/rgbtvg_m3fd_train.pth`
- **RGB**: `../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb/`
- **IR**: `../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/ir/`
- **RGBT**: `../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb/` (用于配对)

### MFAD数据集
- **Label**: `../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_mfad/rgbtvg_mfad_train.pth`
- **RGB**: `../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/rgb/`
- **IR**: `../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/ir/`
- **RGBT**: `../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/mfad/rgb/` (用于配对)

## 📊 输出特性

### RGB/IR模态
- 每个样本生成1张图片 + 1个txt文件
- 图片文件: `mdetr_pred_XXXXXX.jpg`
- 文本文件: `mdetr_pred_XXXXXX.txt`

### RGBT模态 (双图像输出)
- 每个样本生成2张图片 + 1个txt文件
- RGB图片: `mdetr_pred_XXXXXX_rgb.jpg` (彩色)
- IR图片: `mdetr_pred_XXXXXX_ir.jpg` (灰度)
- 文本文件: `mdetr_pred_XXXXXX.txt`

## 🔍 运行示例

### 完整运行日志
```bash
$ bash run_all_mdetr_resnet_tests.sh

🚀 开始MDETR_resnet全面测试...
测试范围：
  - 数据集: flir, m3fd, mfad
  - 模态: rgb, ir, rgbt
  - 总计: 9种组合
========================================

📊 测试 1/9: rgbtvg_flir + rgb
   模型: /home/xijiawen/code/rgbtvg/dataset_and_pretrain_model/result/MDETR_resnet/MDETR_resnet_224_rgb_flir_best.pth
   输出: ./visual_result/mdetr_resnet/rgbtvg_flir/rgb
----------------------------------------
✅ 测试成功: rgbtvg_flir + rgb

📊 测试 2/9: rgbtvg_flir + ir
   模型: /home/xijiawen/code/rgbtvg/dataset_and_pretrain_model/result/MDETR_resnet/MDETR_resnet_224_ir_flir_best.pth
   输出: ./visual_result/mdetr_resnet/rgbtvg_flir/ir
----------------------------------------
✅ 测试成功: rgbtvg_flir + ir

...

🎉 MDETR_resnet全面测试完成！
========================================
📈 测试统计:
   总测试数: 9
   成功: 9
   失败: 0
   耗时: 18分45秒
```

## ⚠️ 注意事项

1. **模型文件**: 确保所有9个模型文件都存在于指定路径
2. **数据路径**: 确保数据集路径正确
3. **GPU内存**: 测试间有2秒间隔避免GPU过载
4. **磁盘空间**: RGBT模态会生成双倍图片，注意磁盘空间
5. **权限**: 确保脚本有执行权限 (`chmod +x`)

## 🛠️ 故障排除

### 模型文件不存在
```bash
❌ 警告: 模型文件不存在: /path/to/model.pth
   跳过此测试...
```
**解决方案**: 检查模型文件路径和文件名

### 数据路径错误
```bash
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/data'
```
**解决方案**: 检查数据集路径配置

### GPU内存不足
```bash
RuntimeError: CUDA out of memory
```
**解决方案**: 
- 减少 `NUM_SAMPLES`
- 增加测试间隔时间
- 使用更大显存的GPU

## 🔄 与CLIP_VG的区别

| 特性 | CLIP_VG | MDETR_resnet |
|------|---------|--------------|
| 模型类型 | ViT-B/16 | ResNet50 |
| 文本处理 | CLIP tokenizer | BERT tokenizer |
| 最大文本长度 | 77 tokens | 20 tokens |
| 输出目录 | `clip_vg/` | `mdetr_resnet/` |
| 文件前缀 | `clip_vg_pred_` | `mdetr_pred_` |

## 📝 扩展用法

### 添加新数据集
在 `visualize_mdetr_resnet.sh` 中添加新的 case 分支：
```bash
"new_dataset")
    LABEL_FILE="path/to/new_dataset_train.pth"
    case $MODALITY in
        "rgb") DATAROOT="path/to/new_dataset/rgb/" ;;
        "ir") DATAROOT="path/to/new_dataset/ir/" ;;
        "rgbt") DATAROOT="path/to/new_dataset/rgb/" ;;
    esac
    ;;
```

### 修改输出目录
```bash
# 自定义输出目录结构
OUTPUT_DIR="./custom_result/mdetr_resnet/${DATASET}/${MODALITY}"
```

---

**创建日期**: 2025-11-12  
**适用版本**: MDETR_resnet v1.0  
**状态**: ✅ 已测试
