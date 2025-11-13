# CLIP_VG BBox格式分析

## 一、训练流程中的bbox处理

### 1. 数据加载 (data_loader.py)
**原始GT格式**：`[x, y, w, h]` - 像素坐标，左上角+宽高

```python
# Line 307-313 in data_loader.py
elif str(self.dataset)[:6] == 'rgbtvg':
    img_file, img_size, bbox, phrase, lighting, scale_cls = self.images[idx]
    bbox_xywh = bbox.copy()  # bbox原始就是 [x, y, w, h] 格式
```

**转换为x1y1x2y2**：
```python
# Line 322-331 in data_loader.py
# 对于rgbtvg数据集，不会进入if分支，因为dataset是'rgbtvg'
# 所以bbox保持原样，但需要注意：
# 如果原始是xywh，会被转换为x1y1x2y2:
bbox = np.array(bbox, dtype=int)
bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]  # x1y1x2y2
```

**实际情况检查**：rgbtvg数据集存储的bbox格式到底是什么？
- 根据代码，rgbtvg不会进入转换分支
- 但是后续transform需要x1y1x2y2格式的输入

### 2. Transform处理 (transforms.py)
**输入**：`box` 是 `[x1, y1, x2, y2]` 格式的像素坐标

```python
# Line 349-357 in transforms.py
if 'box' in input_dict.keys():
    box = input_dict['box']  # x1y1x2y2 格式, 且为 tensor
    box[0], box[2] = box[0] + left, box[2] + left
    box[1], box[3] = box[1] + top, box[3] + top
    h, w = out_img.shape[-2:]
    box = xyxy2xywh(box)  # 转换为 (x_center, y_center, w, h)
    # bbox norm, and xywh
    box = box / torch.tensor([w, h, w, h], dtype=torch.float32)
    input_dict['box'] = box
```

**输出target格式**：`[x_c, y_c, w, h]` - **归一化**的中心点坐标+宽高，范围[0,1]

### 3. 模型输出 (clip_vg.py)
```python
# Line 209 in clip_vg.py
pred_box = self.bbox_embed(vg_hs).sigmoid()
```

**pred_box格式**：`[x_c, y_c, w, h]` - **归一化**的中心点坐标+宽高，范围[0,1]
- 通过sigmoid激活，输出范围[0,1]
- 格式与target一致

### 4. Loss计算 (loss_utils.py)
```python
# Line 251-255 in loss_utils.py
loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')
loss_giou = 1 - torch.diag(generalized_box_iou(
    xywh2xyxy(batch_pred),      # pred: [x_c, y_c, w, h] -> [x1, y1, x2, y2]
    xywh2xyxy(batch_target)     # target: [x_c, y_c, w, h] -> [x1, y1, x2, y2]
))
```

**关键点**：
- `batch_pred`: `[x_c, y_c, w, h]` 归一化，范围[0,1]
- `batch_target`: `[x_c, y_c, w, h]` 归一化，范围[0,1]
- L1 loss直接在归一化的xywh格式上计算
- GIoU需要转换到x1y1x2y2格式，但仍然是归一化坐标

---

## 二、可视化流程中的bbox处理

### 1. 数据加载 (clip_vg_visualize.py)
```python
# Line 373-385 in clip_vg_visualize.py
if str(args.dataset).startswith('rgbtvg'):
    img_filename = item[0]
    img_size = item[1]
    bbox_gt = item[2]  # 原始GT：[x, y, w, h] 像素坐标
    text = item[3]
```

**bbox_gt格式**：`[x, y, w, h]` - **像素坐标**，左上角+宽高

### 2. 模型推理
```python
# Line 437-440 in clip_vg_visualize.py
with torch.no_grad():
    pred_boxes = model(img_nt, text_nt)
bbox = pred_boxes[0].cpu()
```

**bbox格式**：`[x_c, y_c, w, h]` - **归一化**的中心点坐标+宽高，范围[0,1]

### 3. 可视化转换 (当前错误的实现)
```python
# Line 269-278 in clip_vg_visualize.py (当前代码)
# 预测框处理 - 正确
pred_x_center, pred_y_center, pred_bbox_w, pred_bbox_h = pred_bbox
pred_x_min = int((pred_x_center - pred_bbox_w / 2) * w)
pred_y_min = int((pred_y_center - pred_bbox_h / 2) * h)
pred_x_max = int((pred_x_center + pred_bbox_w / 2) * w)
pred_y_max = int((pred_y_center + pred_bbox_h / 2) * h)

# GT框处理 - 错误！
# 当前假设GT是像素坐标的x,y,w,h格式
gt_x_min, gt_y_min, pred_w, pred_h = gt_bbox.astype(int)
gt_x_max = gt_x_min + pred_w
gt_y_max = gt_y_min + pred_h
```

---

## 三、问题分析

### 问题根源
**GT bbox在数据集文件中的实际存储格式**需要确认！

有两种可能：

#### 可能性1：GT存储为 `[x, y, w, h]` 像素坐标
- 这是**原始标注格式**
- data_loader会将其转换为x1y1x2y2后，在transform中转为归一化的xywh
- **可视化时不走transform**，所以仍是像素坐标的xywh

**如果是这种情况，当前可视化代码是正确的！**

#### 可能性2：GT存储为 `[x1, y1, x2, y2]` 像素坐标  
- 这是**转换后的格式**
- data_loader直接使用，transform转换为归一化的xywh
- **可视化时不走transform**，所以是像素坐标的x1y1x2y2

**如果是这种情况，需要修改可视化代码！**

---

## 四、验证方法

添加调试代码查看GT bbox的实际值：

```python
# 在 clip_vg_visualize.py 的 visualize_dataset 函数中
print(f"Sample {sample_idx}:")
print(f"  Image size: {pil_img_original.size}")  # (width, height)
print(f"  GT bbox: {bbox_gt}")
print(f"  GT bbox max value: {np.max(bbox_gt)}")
```

**判断标准**：
- 如果 `max(bbox_gt) > 1.0`：肯定是像素坐标
- 如果 `bbox_gt[2] < bbox_gt[0]` 或 `bbox_gt[3] < bbox_gt[1]`：是x1y1x2y2格式
- 如果 `bbox_gt[2] + bbox_gt[0] < image_width`：可能是xywh格式

---

## 五、正确的可视化实现

### 方案A：如果GT是 `[x, y, w, h]` 像素坐标（当前用户的修改）
```python
# 处理真实框 - GT是像素坐标的xywh格式
gt_x, gt_y, gt_w, gt_h = gt_bbox.astype(int)
gt_x_min = gt_x
gt_y_min = gt_y
gt_x_max = gt_x + gt_w
gt_y_max = gt_y + gt_h
```

### 方案B：如果GT是 `[x1, y1, x2, y2]` 像素坐标
```python
# 处理真实框 - GT是像素坐标的x1y1x2y2格式
gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_bbox.astype(int)
```

---

## 六、结论

根据用户的描述"gt是绝对位置，和图像一个级别的"以及"应该是x,y,w,h的格式"：

**GT bbox在.pth文件中存储为**: `[x, y, w, h]` - 像素坐标的左上角+宽高格式

**用户当前的修改是正确的！**

训练时的处理流程：
1. 读取 `[x, y, w, h]` 像素坐标
2. data_loader转换为 `[x1, y1, x2, y2]` 像素坐标
3. transform转换为 `[x_c, y_c, w, h]` 归一化坐标 [0,1]
4. 模型预测 `[x_c, y_c, w, h]` 归一化坐标 [0,1]
5. Loss计算在归一化的xywh格式上进行

可视化时的处理流程：
1. 读取 `[x, y, w, h]` 像素坐标（不走transform）
2. 模型预测 `[x_c, y_c, w, h]` 归一化坐标 [0,1]
3. 预测框：归一化xywh -> 像素坐标x1y1x2y2
4. GT框：像素坐标xywh -> 像素坐标x1y1x2y2
