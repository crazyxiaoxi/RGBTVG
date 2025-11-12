# MDETR可视化脚本使用指南

## 📝 概述

MDETR可视化脚本支持两种模型版本：
- **MDETR-ResNet**: 使用ResNet作为视觉backbone
- **MDETR-CLIP**: 使用CLIP作为视觉backbone

两种版本使用相同的Python脚本，通过 `--model_type` 参数区分。

## 🚀 快速开始

### 1. MDETR-ResNet版本

```bash
# 编辑配置
vim visualize_scripts/shell_scripts/visualize_mdetr_resnet.sh

# 运行
bash visualize_scripts/shell_scripts/visualize_mdetr_resnet.sh
```

### 2. MDETR-CLIP版本

```bash
# 编辑配置
vim visualize_scripts/shell_scripts/visualize_mdetr_clip.sh

# 运行
bash visualize_scripts/shell_scripts/visualize_mdetr_clip.sh
```

## 📂 输出格式

```
visual_result/
├── mdetr_resnet_rgbtvg_flir_rgb/
│   ├── mdetr_pred_000000.jpg    # ResNet版本预测
│   ├── mdetr_pred_000000.txt
│   └── ...
└── mdetr_clip_rgbtvg_flir_rgb/
    ├── mdetr_pred_000000.jpg    # CLIP版本预测
    ├── mdetr_pred_000000.txt
    └── ...
```

## 🔧 关键参数

### 模型类型参数

```bash
MODEL_TYPE="ResNet"  # 或 "CLIP"
```

这是区分两种版本的关键参数。脚本会根据此参数调用不同的模型实现。

### 配置文件路径

**ResNet版本**:
```bash
MODEL_CHECKPOINT="/path/to/MDETR_ResNet_checkpoint.pth"
```

**CLIP版本**:
```bash
MODEL_CHECKPOINT="/path/to/MDETR_CLIP_checkpoint.pth"
```

## 🎯 模型特点

### MDETR-ResNet
- **Backbone**: ResNet50/101
- **特点**: 传统CNN特征提取
- **适用**: 标准视觉grounding任务

### MDETR-CLIP
- **Backbone**: CLIP视觉编码器
- **特点**: 预训练的视觉-语言对齐
- **适用**: 需要更强语义理解的任务

## 📊 与其他模型对比

| 模型 | Backbone | 文本处理 | 返回格式 |
|------|----------|----------|----------|
| MDETR-ResNet | ResNet | BERT | pred_box |
| MDETR-CLIP | CLIP | BERT | pred_box |
| TransVG | ResNet | BERT | pred_box |
| MMCA | ResNet | BERT | pred_box |
| HiVG | CLIP | CLIP | 5元组 |

## 🔍 工作原理

### 输入处理

```python
# 图像: NestedTensor(img_tensor, img_mask)
# 文本: NestedTensor(token_ids, attention_mask)
```

### 模型调用

```python
pred_boxes = model(img_nt, text_nt)  # 直接返回pred_box
```

### 两种版本的区别

主要在 `build_model()` 时根据 `args.model_type` 选择：
- `model_type='ResNet'` → `DynamicMDETR_ResNet`
- `model_type='CLIP'` → `DynamicMDETR_CLIP`

## 💡 使用技巧

### 1. 比较两种版本

```bash
# 使用相同的样本
START_IDX=0
NUM_SAMPLES=100

# 分别运行两个版本
bash visualize_scripts/shell_scripts/visualize_mdetr_resnet.sh
bash visualize_scripts/shell_scripts/visualize_mdetr_clip.sh

# 对比输出目录
ls visual_result/mdetr_resnet_rgbtvg_flir_rgb/
ls visual_result/mdetr_clip_rgbtvg_flir_rgb/
```

### 2. 切换数据集

```bash
# M3FD数据集
DATASET="rgbtvg_m3fd"
LABEL_FILE="../dataset_and_pretrain_model/datasets/VG/ref_data_shuffled/rgbtvg_m3fd/rgbtvg_m3fd_train.pth"
DATAROOT="../dataset_and_pretrain_model/datasets/VG/image_data/rgbtvg/rgbtvg-images/m3fd/rgb/"
```

### 3. RGBT模态

```bash
MODALITY="rgbt"
# DATAROOT仍指向RGB，脚本会自动查找IR图像
```

## ⚠️ 注意事项

1. **Model Type必须正确**: 确保 `MODEL_TYPE` 与实际checkpoint匹配
2. **Checkpoint配置**: 脚本会自动使用checkpoint中保存的模型配置
3. **文本Tokenization**: 使用BERT tokenizer处理文本
4. **数组连续性**: 已处理OpenCV兼容性问题

## 🐛 常见问题

### 1. Model Type不匹配

**错误**: 加载checkpoint后模型结构不对

**解决**: 检查 `MODEL_TYPE` 参数是否与checkpoint训练时一致

### 2. 找不到checkpoint

**错误**: `FileNotFoundError`

**解决**: 修改 `MODEL_CHECKPOINT` 为实际路径

### 3. GPU内存不足

**解决**: 
- 减少 `NUM_SAMPLES`
- 使用更小的 `IMSIZE`

## 📚 相关文档

- [TransVG使用指南](./TRANSVG_VISUALIZATION_GUIDE.md)
- [HiVG修复总结](./HIVG_VISUALIZATION_FIX_SUMMARY.md)
- [总体README](./README.md)

## 🎨 输出示例

**图片**: 绿色bbox框  
**文本**: UTF-8编码的referring expression

---

**创建日期**: 2025-11-12  
**状态**: ✅ 已完成
