# CLIP_VG BBox格式处理流程对比总结

## 📊 完整对比表

| 阶段 | 训练流程 | 可视化流程 |
|------|----------|-----------|
| **数据加载** | 从.pth读取：`[x, y, w, h]` 像素坐标 | 从.pth读取：`[x, y, w, h]` 像素坐标 |
| **预处理** | data_loader转为：`[x1, y1, x2, y2]` 像素坐标 | 无预处理，保持原格式 |
| **Transform** | transform转为：`[x_c, y_c, w, h]` 归一化[0,1] | 不经过transform |
| **模型输入/GT** | GT = `[x_c, y_c, w, h]` 归一化[0,1] | GT = `[x, y, w, h]` 像素坐标（未变） |
| **模型输出** | Pred = `[x_c, y_c, w, h]` 归一化[0,1] | Pred = `[x_c, y_c, w, h]` 归一化[0,1] |
| **Loss/可视化** | 直接在归一化坐标上计算loss | 需转换为像素坐标绘制bbox |

---

## 🔍 详细流程分析

### 1️⃣ 训练流程（从pred_box到计算loss）

#### Step 1: 数据集存储格式
```python
# rgbtvg_xxx_train.pth 中的数据项
item = [img_file, img_size, bbox, phrase, lighting, scale_cls]
# bbox = [x, y, w, h]  # 像素坐标，左上角+宽高
# 例如：[100, 150, 50, 80] 表示从(100,150)开始，宽50，高80
```

#### Step 2: DataLoader处理
```python
# datasets/data_loader.py Line 307-331
bbox_ori = bbox.copy()  # [x, y, w, h] 像素坐标

# 对于rgbtvg数据集，转换为x1y1x2y2格式
if not (self.dataset == 'referit' or self.dataset == 'flickr'):
    bbox = np.array(bbox, dtype=int)
    bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
    # 转换后：[x, y, x+w, y+h] = [x1, y1, x2, y2]
    # 例如：[100, 150, 50, 80] -> [100, 150, 150, 230]
```

#### Step 3: Transform处理
```python
# datasets/transforms.py Line 349-357
box = input_dict['box']  # [x1, y1, x2, y2] 像素坐标

# 先应用各种变换（crop, pad等），然后：
box = xyxy2xywh(box)  # 转换为中心点+宽高格式
# [x1, y1, x2, y2] -> [x_c, y_c, w, h]
# 其中 x_c = (x1+x2)/2, y_c = (y1+y2)/2, w = x2-x1, h = y2-y1
# 例如：[100, 150, 150, 230] -> [125, 190, 50, 80]

box = box / torch.tensor([w, h, w, h], dtype=torch.float32)
# 归一化到[0,1]
# 例如：图像尺寸640x480
# [125, 190, 50, 80] -> [125/640, 190/480, 50/640, 80/480]
#                     = [0.195, 0.396, 0.078, 0.167]
```

#### Step 4: 模型前向传播
```python
# models/clip_vg.py Line 207-211
vg_hs = self.vl_transformer(vl_src, vl_mask, vl_pos)
vg_hs = vg_hs[0]
pred_box = self.bbox_embed(vg_hs).sigmoid()
# pred_box: [x_c, y_c, w, h] 归一化坐标，范围[0,1]
# 例如：[0.198, 0.385, 0.082, 0.171]
```

#### Step 5: Loss计算
```python
# utils/loss_utils.py Line 251-255
# batch_pred: [x_c, y_c, w, h] 归一化[0,1]
# batch_target: [x_c, y_c, w, h] 归一化[0,1]

# L1 Loss: 直接在归一化坐标上计算
loss_bbox = F.l1_loss(batch_pred, batch_target, reduction='none')

# GIoU Loss: 需要先转换到xyxy格式
loss_giou = 1 - torch.diag(generalized_box_iou(
    xywh2xyxy(batch_pred),      # [x_c, y_c, w, h] -> [x1, y1, x2, y2]
    xywh2xyxy(batch_target)     # [x_c, y_c, w, h] -> [x1, y1, x2, y2]
))
# 注意：转换后仍然是归一化坐标！
```

**关键点**：
- ✅ 预测和GT都是归一化的 `[x_c, y_c, w, h]` 格式
- ✅ Loss在同一坐标系下计算，公平对比

---

### 2️⃣ 可视化流程（从pred_box到绘制bbox）

#### Step 1: 数据集读取
```python
# visualize_scripts/clip_vg_visualize.py Line 373-377
if str(args.dataset).startswith('rgbtvg'):
    img_filename = item[0]
    img_size = item[1]
    bbox_gt = item[2]  # [x, y, w, h] 像素坐标，左上角+宽高
    text = item[3]

# ⚠️ 注意：bbox_gt直接从.pth读取，不经过data_loader的转换！
# 例如：[100, 150, 50, 80]
```

#### Step 2: 图像处理
```python
# 图像经过transform，但bbox_gt不变
result = process_image(args, img_path, text, transform)
img_tensor, img_mask, pil_img_original = result

# pil_img_original: 原始RGB图像，用于可视化
# 尺寸例如：640x480 (width x height)
```

#### Step 3: 模型推理
```python
# Line 437-440
with torch.no_grad():
    pred_boxes = model(img_nt, text_nt)
bbox = pred_boxes[0].cpu()

# bbox: [x_c, y_c, w, h] 归一化坐标，范围[0,1]
# 例如：[0.198, 0.385, 0.082, 0.171]
```

#### Step 4: 坐标转换用于可视化
```python
# Line 267-285 (预测框转换)
h, w = img_np.shape[:2]  # 例如：h=480, w=640

# 预测框：归一化的(x_c, y_c, w, h) -> 像素坐标的(x1, y1, x2, y2)
pred_x_center, pred_y_center, pred_bbox_w, pred_bbox_h = pred_bbox
# 例如：0.198, 0.385, 0.082, 0.171

pred_x_min = int((pred_x_center - pred_bbox_w / 2) * w)
pred_y_min = int((pred_y_center - pred_bbox_h / 2) * h)
pred_x_max = int((pred_x_center + pred_bbox_w / 2) * w)
pred_y_max = int((pred_y_center + pred_bbox_h / 2) * h)
# 计算过程：
# x_min = (0.198 - 0.082/2) * 640 = 0.157 * 640 = 100
# y_min = (0.385 - 0.171/2) * 480 = 0.300 * 480 = 144
# x_max = (0.198 + 0.082/2) * 640 = 0.239 * 640 = 153
# y_max = (0.385 + 0.171/2) * 480 = 0.471 * 480 = 226
# 结果：[100, 144, 153, 226]

# Line 286-298 (GT框转换)
# GT框：像素坐标的(x, y, w, h) -> 像素坐标的(x1, y1, x2, y2)
gt_x, gt_y, gt_w, gt_h = gt_bbox.astype(int)
# 例如：100, 150, 50, 80

gt_x_min = gt_x          # 100
gt_y_min = gt_y          # 150
gt_x_max = gt_x + gt_w   # 100 + 50 = 150
gt_y_max = gt_y + gt_h   # 150 + 80 = 230
# 结果：[100, 150, 150, 230]
```

#### Step 5: 绘制bbox
```python
# Line 311-313
cv2.rectangle(vis_img_rgb, 
              (pred_x_min, pred_y_min), (pred_x_max, pred_y_max), 
              (0, 255, 0), 2)  # 绿色预测框

cv2.rectangle(vis_img_rgb, 
              (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), 
              (0, 0, 255), 2)  # 红色真实框
```

---

## ⚠️ 关键区别

| 项目 | 训练 | 可视化 |
|------|------|--------|
| **GT格式** | 归一化 `[x_c, y_c, w, h]` | 像素 `[x, y, w, h]` |
| **Pred格式** | 归一化 `[x_c, y_c, w, h]` | 归一化 `[x_c, y_c, w, h]` |
| **坐标系统** | 归一化[0,1] | 像素坐标 |
| **数据流** | 经过transform归一化 | 不经过transform |

---

## ✅ 当前实现验证

### 预测框处理 ✅ 正确
```python
# 归一化的中心点格式 -> 像素坐标的角点格式
pred_x_min = int((pred_x_center - pred_bbox_w / 2) * w)
pred_y_min = int((pred_y_center - pred_bbox_h / 2) * h)
pred_x_max = int((pred_x_center + pred_bbox_w / 2) * w)
pred_y_max = int((pred_y_center + pred_bbox_h / 2) * h)
```

### GT框处理 ✅ 正确
```python
# 像素坐标的左上角+宽高 -> 像素坐标的角点格式
gt_x, gt_y, gt_w, gt_h = gt_bbox.astype(int)
gt_x_min = gt_x
gt_y_min = gt_y
gt_x_max = gt_x + gt_w
gt_y_max = gt_y + gt_h
```

---

## 🎯 总结

1. **训练时**：
   - GT和Pred都是**归一化的** `(x_center, y_center, w, h)` 格式
   - Loss在归一化坐标系下计算，坐标系一致

2. **可视化时**：
   - GT是**像素坐标**的 `(x, y, w, h)` 格式（左上角+宽高）
   - Pred是**归一化**的 `(x_center, y_center, w, h)` 格式
   - 需要分别转换为像素坐标的 `(x1, y1, x2, y2)` 才能绘制

3. **当前实现**：✅ **完全正确**
   - 预测框：归一化中心点格式 -> 像素角点格式
   - GT框：像素左上角+宽高 -> 像素角点格式
   - 两者都正确转换为OpenCV绘制所需的格式

4. **颜色标注**：
   - 🟢 绿色：模型预测框
   - 🔴 红色：真实标注框
