"""
公共可视化工具模块
包含所有模型通用的图像处理和可视化保存函数
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path


def process_image(args, img_path, text, transform):
    """处理单张图像，返回transform后的图像用于模型推理和原始图像用于可视化
    
    Args:
        args: 参数对象，包含modality等配置
        img_path: 图像路径
        text: 文本描述
        transform: 图像变换函数
        
    Returns:
        对于RGBT模态: (img_tensor, img_mask, pil_img_original, pil_img_ir)
        对于其他模态: (img_tensor, img_mask, pil_img_original)
    """
    pil_img_original = None  # 保存原始RGB图像用于可视化
    pil_img_ir = None  # 保存原始IR图像用于可视化
    
    if args.modality == 'rgbt':
        # 尝试自动配对RGB和IR图像
        if '/rgb/' in img_path:
            rgb_path = img_path
            ir_path = img_path.replace('/rgb/', '/ir/')
        elif '/ir/' in img_path:
            ir_path = img_path
            rgb_path = img_path.replace('/ir/', '/rgb/')
        else:
            rgb_path = img_path
            ir_path = img_path.replace('rgb', 'ir')
        
        # 检查文件是否存在
        if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
            print(f"Warning: RGB or IR image not found: {rgb_path}, {ir_path}")
            return None, None, None, None
        
        # 加载RGB和IR图像，完全模仿数据加载器的逻辑
        img_rgb_path = rgb_path
        img_ir_path = ir_path
        img_rgb = Image.open(img_rgb_path).convert("RGB")
        img_ir = Image.open(img_ir_path)
        pil_img_original = img_rgb.copy()
        pil_img_ir = img_ir.copy()

        # 与数据加载器完全一致的处理
        np_rgb = np.array(img_rgb)
        np_ir = np.array(img_ir)
        if np_ir.shape[-1] == 3:
            np_ir = np_ir[..., 0]
        np_ir = np.expand_dims(np_ir, axis=-1)
        np_combined = np.concatenate([np_rgb, np_ir], axis=-1)
        img = Image.fromarray(np_combined)

        # 获取图像尺寸
        w, h = img.size
        full_box_xyxy = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], dtype=torch.float32)

        # 使用transform处理RGBT图像
        input_dict = {'img': img, 'box': full_box_xyxy, 'text': text}
        input_dict = transform(input_dict)

        return input_dict['img'], input_dict['mask'], pil_img_original, pil_img_ir
        
    else:
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return None, None, None
        # IR图像转换为RGB（3通道），与dataloader保持一致
        pil_img = Image.open(img_path).convert('RGB')
        
        # 保存原始图像用于可视化
        pil_img_original = pil_img.copy()
        
        # 获取图像尺寸
        w, h = pil_img.size
        full_box_xyxy = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], dtype=torch.float32)
        
        # 应用变换
        input_dict = {'img': pil_img, 'box': full_box_xyxy, 'text': text}
        input_dict = transform(input_dict)

        return input_dict['img'], input_dict['mask'], pil_img_original


def save_gt_visualization(args, pil_img_original, pil_img_ir, text, gt_bbox, sample_idx, output_dir, model_name="model"):
    """保存GT可视化结果（仅显示真实框）
    对于RGBT模态：保存RGB图+GT框、IR图+GT框、1个txt文件、1张原图（不带框）
    
    Args:
        args: 参数对象，包含modality、dataset等配置
        pil_img_original: 原始RGB图像
        pil_img_ir: 原始IR图像（可选）
        text: 文本描述
        gt_bbox: 真实边界框
        sample_idx: 样本索引
        output_dir: 输出目录
        model_name: 模型名称，用于文件命名
        
    Returns:
        str: RGB图像保存路径
    """
    img_np = np.array(pil_img_original)
    
    h, w = img_np.shape[:2]

    if isinstance(gt_bbox, torch.Tensor):
        gt_bbox = gt_bbox.cpu().numpy()
    elif isinstance(gt_bbox, list):
        gt_bbox = np.array(gt_bbox)
    
    if len(gt_bbox) == 4:
        gt_x, gt_y, gt_w, gt_h = gt_bbox.astype(int)
        gt_x_min = gt_x
        gt_y_min = gt_y
        gt_x_max = gt_x + gt_w
        gt_y_max = gt_y + gt_h
    else:
        print(f"Warning: Unexpected gt_bbox format: {gt_bbox}")
        gt_x_min = gt_y_min = gt_x_max = gt_y_max = 0
    
    gt_x_min = max(0, min(gt_x_min, w - 1))
    gt_y_min = max(0, min(gt_y_min, h - 1))
    gt_x_max = max(0, min(gt_x_max, w - 1))
    gt_y_max = max(0, min(gt_y_max, h - 1))
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # 3. 保存原图（不带框）- 只对RGBT模态保存
    original_path = os.path.join(output_dir, f"{model_name}_{sample_idx:06d}_original.jpg")
    cv2.imwrite(original_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    # 1. 保存RGB图 + GT框
    vis_img_rgb = np.ascontiguousarray(img_np)
    cv2.rectangle(vis_img_rgb, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (0, 0, 255), 2)  # 红色真实框
    rgb_path = os.path.join(output_dir, f"{model_name}_{sample_idx:06d}_rgb.jpg")
    cv2.imwrite(rgb_path, cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR))
    
    # 2. 对于RGBT模态，保存IR图 + GT框
    if hasattr(args, 'modality') and args.modality == 'rgbt' and pil_img_ir is not None:
        img_ir_np = np.array(pil_img_ir)

        if img_ir_np.ndim == 2:
            img_ir_np = np.stack([img_ir_np] * 3, axis=-1)
        elif img_ir_np.ndim == 3 and img_ir_np.shape[2] == 1:
            img_ir_np = np.repeat(img_ir_np, 3, axis=2)
        
        vis_img_ir = np.ascontiguousarray(img_ir_np)
        cv2.rectangle(vis_img_ir, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (0, 0, 255), 2)  # 红色真实框
        ir_path = os.path.join(output_dir, f"{model_name}_{sample_idx:06d}_ir.jpg")
        cv2.imwrite(ir_path, cv2.cvtColor(vis_img_ir, cv2.COLOR_RGB2BGR))
        

    
    # 4. 保存文本文件
    txt_path = os.path.join(output_dir, f"{model_name}_{sample_idx:06d}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return rgb_path


def save_visualization(args, pil_img_original, pil_img_ir, text, pred_bbox, gt_bbox, sample_idx, output_dir, model_name="model"):
    """保存可视化结果，分别保存预测框和真实框的图像
    
    Args:
        args: 参数对象，包含modality、dataset等配置
        pil_img_original: 原始RGB图像
        pil_img_ir: 原始IR图像（可选）
        text: 文本描述
        pred_bbox: 预测的边界框
        gt_bbox: 真实边界框
        sample_idx: 样本索引
        output_dir: 输出目录
        model_name: 模型名称，用于文件命名
        
    Returns:
        tuple: (pred_rgb_path, gt_rgb_path)
    """
    # 保存预测框图像
    pred_rgb_path = save_pred_visualization(args, pil_img_original, pil_img_ir, text, pred_bbox, sample_idx, output_dir, model_name)
    
    # 保存GT框图像
    gt_rgb_path = save_gt_visualization(args, pil_img_original, pil_img_ir, text, gt_bbox, sample_idx, output_dir, model_name)
    
    return pred_rgb_path, gt_rgb_path


def save_pred_visualization(args, pil_img_original, pil_img_ir, text, pred_bbox, sample_idx, output_dir, model_name="model"):
    """保存预测可视化结果（仅显示预测框）
    
    Args:
        args: 参数对象，包含modality、dataset等配置
        pil_img_original: 原始RGB图像
        pil_img_ir: 原始IR图像（可选）
        text: 文本描述
        pred_bbox: 预测的边界框
        sample_idx: 样本索引
        output_dir: 输出目录
        model_name: 模型名称，用于文件命名
        
    Returns:
        str: RGB图像保存路径
    """
    # 直接使用原始图像，不需要反归一化
    img_np = np.array(pil_img_original)
    
    # 转换bbox到像素坐标
    h, w = img_np.shape[:2]
    
    # 处理预测框 - 模型输出是sigmoid后的归一化坐标(x_center, y_center, w, h)，范围[0,1]
    if isinstance(pred_bbox, torch.Tensor):
        pred_bbox = pred_bbox.cpu().numpy()
    
    # 预测框格式：归一化的(x_center, y_center, w, h)
    # 需要转换为像素坐标的(x_min, y_min, x_max, y_max)用于绘制
    pred_x_center, pred_y_center, pred_bbox_w, pred_bbox_h = pred_bbox
    
    # 基础转换：归一化坐标转像素坐标
    pred_x_min = int((pred_x_center - pred_bbox_w / 2) * w)
    pred_y_min = int((pred_y_center - pred_bbox_h / 2) * w)  # 注意：这里用w是正确的
    pred_x_max = int((pred_x_center + pred_bbox_w / 2) * w)
    pred_y_max = int((pred_y_center + pred_bbox_h / 2) * w)  # 注意：这里用w是正确的
    
    # 数据集特定的Y轴偏移调整
    if hasattr(args, 'dataset'):
        if args.dataset == 'rgbtvg_mfad':
            pred_y_min -= 160
            pred_y_max -= 160
        elif args.dataset == 'rgbtvg_m3fd':
            pred_y_min -= 128
            pred_y_max -= 128
        elif args.dataset == 'rgbtvg_flir':
            pred_y_min -= 64
            pred_y_max -= 64
    
    # 限制预测框在图像范围内
    pred_x_min = max(0, min(pred_x_min, w - 1))
    pred_y_min = max(0, min(pred_y_min, h - 1))
    pred_x_max = max(0, min(pred_x_max, w - 1))
    pred_y_max = max(0, min(pred_y_max, h - 1))
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存RGB图像（仅预测框）
    vis_img_rgb = np.ascontiguousarray(img_np)
    cv2.rectangle(vis_img_rgb, (pred_x_min, pred_y_min), (pred_x_max, pred_y_max), (0, 255, 0), 2)  # 绿色预测框
    rgb_path = os.path.join(output_dir, f"{model_name}_pred_{sample_idx:06d}_rgb.jpg")
    cv2.imwrite(rgb_path, cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR))
    
    # 如果是RGBT模态，还要保存IR图像（仅预测框）
    if hasattr(args, 'modality') and args.modality == 'rgbt' and pil_img_ir is not None:
        img_ir_np = np.array(pil_img_ir)
        # 如果IR是单通道，转换为3通道用于可视化
        if img_ir_np.ndim == 2:
            img_ir_np = np.stack([img_ir_np] * 3, axis=-1)
        elif img_ir_np.ndim == 3 and img_ir_np.shape[2] == 1:
            img_ir_np = np.repeat(img_ir_np, 3, axis=2)
        
        vis_img_ir = np.ascontiguousarray(img_ir_np)
        cv2.rectangle(vis_img_ir, (pred_x_min, pred_y_min), (pred_x_max, pred_y_max), (0, 255, 0), 2)  # 绿色预测框
        ir_path = os.path.join(output_dir, f"{model_name}_pred_{sample_idx:06d}_ir.jpg")
        cv2.imwrite(ir_path, cv2.cvtColor(vis_img_ir, cv2.COLOR_RGB2BGR))
    
    # 保存文本到txt文件
    txt_path = os.path.join(output_dir, f"{model_name}_pred_{sample_idx:06d}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return rgb_path


def load_dataset(label_file):
    """加载数据集文件
    
    Args:
        label_file: 数据集标签文件路径
        
    Returns:
        list: 数据集列表
    """
    import torch
    
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Dataset file not found: {label_file}")
    
    print(f"Loading dataset from: {label_file}")
    data = torch.load(label_file, map_location='cpu')
    print(f"Total samples in dataset: {len(data)}")
    return data


def save_combined_pred_visualization(args, pil_img_original, pil_img_ir, predictions, img_filename, output_dir, model_name):
    """保存合并的预测可视化结果（每个图片以文件夹形式存储）"""
    import torch
    import numpy as np
    import cv2
    from pathlib import Path
    
    img_np = np.array(pil_img_original)
    h, w = img_np.shape[:2]
    
    # 为每个图片创建单独的文件夹
    img_base_name = Path(img_filename).stem
    img_folder = os.path.join(output_dir, img_base_name)
    Path(img_folder).mkdir(parents=True, exist_ok=True)
    
    # 生成不同颜色用于区分不同的预测框
    colors = [
        (0, 255, 0),    # 绿色 - 预测框用绿色
        (255, 0, 0),    # 红色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
        (255, 128, 0),  # 橙色
        (128, 0, 255),  # 紫罗兰
    ]
    
    # 1. 保存RGB原图（不带框）
    rgb_original_path = os.path.join(img_folder, "rgb_original.jpg")
    cv2.imwrite(rgb_original_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    
    # 2. 对于RGBT模态，保存IR原图（不带框）
    if hasattr(args, 'modality') and args.modality == 'rgbt' and pil_img_ir is not None:
        img_ir_np = np.array(pil_img_ir)
        
        # 保存IR原图
        ir_original_path = os.path.join(img_folder, "ir_original.jpg")
        if img_ir_np.ndim == 2:
            cv2.imwrite(ir_original_path, img_ir_np)
        else:
            cv2.imwrite(ir_original_path, cv2.cvtColor(img_ir_np, cv2.COLOR_RGB2BGR))
    
    # 3. 保存RGB图 + 所有预测框
    vis_img_rgb = np.ascontiguousarray(img_np.copy())
    
    # 收集所有文本
    all_texts = []
    
    for i, pred in enumerate(predictions):
        bbox_pred = pred['bbox']
        text = pred['text']
        sample_idx = pred['sample_idx']
        
        # 处理bbox
        if isinstance(bbox_pred, torch.Tensor):
            bbox_pred = bbox_pred.cpu().numpy()
        elif isinstance(bbox_pred, list):
            bbox_pred = np.array(bbox_pred)
        
        if len(bbox_pred) == 4:
            # 假设是xywh格式，转换为xyxy
            x_center, y_center, bbox_w, bbox_h = bbox_pred
            x_min = int((x_center - bbox_w / 2) * w)
            y_min = int((y_center - bbox_h / 2) * h)
            x_max = int((x_center + bbox_w / 2) * w)
            y_max = int((y_center + bbox_h / 2) * h)
        else:
            print(f"Warning: Unexpected pred_bbox format: {bbox_pred}")
            continue
        
        # 限制在图像范围内
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(0, min(x_max, w - 1))
        y_max = max(0, min(y_max, h - 1))
        
        # 选择颜色
        color = colors[i % len(colors)]
        
        # 画框
        cv2.rectangle(vis_img_rgb, (x_min, y_min), (x_max, y_max), color, 2)
        
        # 添加标签（框的编号）
        cv2.putText(vis_img_rgb, f"{i+1}", (x_min, y_min-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 收集文本
        all_texts.append(f"{i+1}. {text}")
    
    # 保存RGB图 + 预测框
    rgb_pred_path = os.path.join(img_folder, f"rgb_with_{model_name}_pred.jpg")
    cv2.imwrite(rgb_pred_path, cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR))
    
    # 4. 对于RGBT模态，保存IR图 + 所有预测框
    if hasattr(args, 'modality') and args.modality == 'rgbt' and pil_img_ir is not None:
        img_ir_np = np.array(pil_img_ir)
        
        if img_ir_np.ndim == 2:
            img_ir_np = np.stack([img_ir_np] * 3, axis=-1)
        elif img_ir_np.ndim == 3 and img_ir_np.shape[2] == 1:
            img_ir_np = np.repeat(img_ir_np, 3, axis=2)
        
        vis_img_ir = np.ascontiguousarray(img_ir_np.copy())
        
        # 画所有预测框到IR图上
        for i, pred in enumerate(predictions):
            bbox_pred = pred['bbox']
            
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.cpu().numpy()
            elif isinstance(bbox_pred, list):
                bbox_pred = np.array(bbox_pred)
            
            if len(bbox_pred) == 4:
                x_center, y_center, bbox_w, bbox_h = bbox_pred
                x_min = max(0, min(int((x_center - bbox_w / 2) * w), w - 1))
                y_min = max(0, min(int((y_center - bbox_h / 2) * h), h - 1))
                x_max = max(0, min(int((x_center + bbox_w / 2) * w), w - 1))
                y_max = max(0, min(int((y_center + bbox_h / 2) * h), h - 1))
                
                color = colors[i % len(colors)]
                cv2.rectangle(vis_img_ir, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(vis_img_ir, f"{i+1}", (x_min, y_min-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        ir_pred_path = os.path.join(img_folder, f"ir_with_{model_name}_pred.jpg")
        cv2.imwrite(ir_pred_path, cv2.cvtColor(vis_img_ir, cv2.COLOR_RGB2BGR))
    
    # 5. 保存合并的文本文件
    txt_path = os.path.join(img_folder, f"{model_name}_predictions.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Image: {img_filename}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total predictions: {len(predictions)}\n")
        f.write("=" * 50 + "\n\n")
        for text_line in all_texts:
            f.write(text_line + "\n\n")
    
    return rgb_pred_path


def generate_prediction_statistics(output_dir, prediction_stats, dataset, modality, model_name):
    """生成预测统计报告"""
    from pathlib import Path
    
    # 排序：按预测数量降序排列
    prediction_stats.sort(key=lambda x: x['predictions'], reverse=True)
    
    # 计算统计信息
    total_images = len(prediction_stats)
    total_predictions = sum(item['predictions'] for item in prediction_stats)
    avg_predictions = total_predictions / total_images if total_images > 0 else 0
    
    # 统计预测数量分布
    prediction_counts = {}
    for item in prediction_stats:
        count = item['predictions']
        prediction_counts[count] = prediction_counts.get(count, 0) + 1
    
    # 保存统计报告
    stats_path = os.path.join(output_dir, "prediction_statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{model_name.upper()} PREDICTION STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Modality: {modality}\n")
        f.write(f"Generated: {Path().absolute()}\n\n")
        
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Total Predictions: {total_predictions}\n")
        f.write(f"Average Predictions per Image: {avg_predictions:.2f}\n\n")
        
        f.write("PREDICTION COUNT DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        for count in sorted(prediction_counts.keys()):
            images_with_count = prediction_counts[count]
            percentage = (images_with_count / total_images) * 100
            f.write(f"{count} predictions: {images_with_count} images ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("DETAILED LIST (sorted by prediction count):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Image':<50} {'Predictions':<12}\n")
        f.write("-" * 80 + "\n")
        
        for i, item in enumerate(prediction_stats, 1):
            f.write(f"{i:<6} {item['image']:<50} {item['predictions']:<12}\n")
    
    # 同时生成CSV格式的统计文件
    csv_path = os.path.join(output_dir, "prediction_statistics.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Rank,Image,Predictions\n")
        for i, item in enumerate(prediction_stats, 1):
            f.write(f"{i},{item['image']},{item['predictions']}\n")
    
    print(f"\n📊 {model_name.upper()} Prediction Statistics:")
    print(f"   Total Images: {total_images}")
    print(f"   Total Predictions: {total_predictions}")
    print(f"   Average per Image: {avg_predictions:.2f}")
    print(f"   Max Predictions: {max(item['predictions'] for item in prediction_stats) if prediction_stats else 0}")
    print(f"   Min Predictions: {min(item['predictions'] for item in prediction_stats) if prediction_stats else 0}")