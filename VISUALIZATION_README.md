# OneRef 可视化功能使用说明

## 概述

已为 OneRef 模型添加了可视化功能，可以在评估时生成包含预测边界框、真实边界框和分割掩码的可视化结果。

## 使用方法

### 1. 基本使用

在运行 `oneref_eval.py` 时，添加 `--visualize` 参数即可启用可视化：

```bash
python oneref_eval.py \
    --visualize \
    --task grounding \
    --eval_model <模型路径> \
    --eval_set test \
    --dataset <数据集名称> \
    --modality <模态> \
    --imsize 224 \
    --batch_size 1 \
    --其他参数...
```

### 2. 必需参数

- `--task`: 任务类型，对于 OneRef 模型应设置为 `grounding`（必需）
- `--eval_model`: 模型检查点路径（必需）

### 3. 可视化参数

- `--visualize`: 启用可视化功能（可选）
- `--visual_output_dir`: 可视化结果保存目录（默认：`./visual_result/oneref`）
- `--visualize_num_samples`: 要可视化的样本数量（默认：100，设置为0表示可视化所有样本）

### 4. 示例

```bash
# 可视化前2个样本
python oneref_eval.py \
    --visualize \
    --task grounding \
    --max_query_len 64 \
    --model beit3_base_patch16_224 \
    --visualize_num_samples 2 \
    --visual_output_dir ./visual_result/oneref_test \
    --sentencepiece_model ../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm \
    --eval_model /root/autodl-tmp/cvpr/xjw_codes/output_training/ONEREF_base_rec_224_rgb/rgbtvg_mfad/best_checkpoint.pth \
    --eval_set test \
    --dataset rgbtvg_mfad \
    --modality rgb \
    --imsize 224 \
    --batch_size 1

# 可视化所有样本
python oneref_eval.py \
    --visualize \
    --task grounding \
    --visualize_num_samples 0 \
    --visual_output_dir ./visual_result/oneref_all \
    --eval_model ./checkpoints/best_model.pth \
    --eval_set test \
    --dataset rgbtvg_mfad \
    --modality rgbt \
    --imsize 224 \
    --batch_size 1
```

## 可视化结果说明

生成的可视化图像包含以下内容：

1. **预测边界框**（绿色）：模型预测的目标边界框
2. **真实边界框**（红色）：标注的真实目标边界框
3. **预测分割掩码**（绿色半透明覆盖）：模型预测的分割掩码
4. **真实分割掩码**（红色轮廓）：标注的真实分割掩码
5. **文本描述**：输入的自然语言描述
6. **图例**：说明预测和真实标注的颜色

## 输出文件

可视化结果保存在指定的 `visual_output_dir` 目录中，文件命名格式为：
```
oneref_result_000000.jpg
oneref_result_000001.jpg
...
```

## 注意事项

1. 可视化功能仅在评估时可用，不会影响训练过程
2. 可视化功能仅在主进程（main process）中执行，避免分布式训练时的重复保存
3. 如果模型没有启用分割掩码（`use_mask_loss=False`），则只会显示边界框
4. 可视化会增加一定的计算开销，建议在需要时再启用

## 代码结构

- `utils/oneref_visual_utils.py`: OneRef 可视化工具模块
- `engine.py`: 评估函数中集成了可视化调用
- `oneref_eval.py`: 评估脚本，添加了可视化参数

## 扩展其他模型

目前可视化功能仅适配了 OneRef 模型。如需为其他模型添加可视化功能，可以：

1. 参考 `utils/oneref_visual_utils.py` 创建对应的可视化工具
2. 在 `engine.py` 的 `evaluate` 函数中添加对应的可视化调用
3. 在对应的评估脚本中添加可视化参数

