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
    
    vis_img_rgb = np.ascontiguousarray(img_np)
    cv2.rectangle(vis_img_rgb, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (0, 0, 255), 2)  # 红色真实框
    rgb_path = os.path.join(output_dir, f"{model_name}_gt_{sample_idx:06d}_rgb.jpg")
    cv2.imwrite(rgb_path, cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR))
    
    if hasattr(args, 'modality') and args.modality == 'rgbt' and pil_img_ir is not None:
        img_ir_np = np.array(pil_img_ir)

        if img_ir_np.ndim == 2:
            img_ir_np = np.stack([img_ir_np] * 3, axis=-1)
        elif img_ir_np.ndim == 3 and img_ir_np.shape[2] == 1:
            img_ir_np = np.repeat(img_ir_np, 3, axis=2)
        
        vis_img_ir = np.ascontiguousarray(img_ir_np)
        cv2.rectangle(vis_img_ir, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (0, 0, 255), 2)  # 红色真实框
        ir_path = os.path.join(output_dir, f"{model_name}_gt_{sample_idx:06d}_ir.jpg")
        cv2.imwrite(ir_path, cv2.cvtColor(vis_img_ir, cv2.COLOR_RGB2BGR))
    
    txt_path = os.path.join(output_dir, f"{model_name}_gt_{sample_idx:06d}.txt")
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