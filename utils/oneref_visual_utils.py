"""
OneRef模型可视化工具
用于可视化OneRef模型的预测结果，包括边界框和分割掩码
"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from utils.box_utils import xywh2xyxy


def get_normalization_params(args):
    """根据数据集和模态获取归一化参数"""
    if args.modality == 'rgbt':
        if args.dataset == 'rgbtvg_flir':
            mean = [0.631, 0.6401, 0.632, 0.5337]
            std = [0.2152, 0.227, 0.2439, 0.2562]
        elif args.dataset == 'rgbtvg_m3fd':
            mean = [0.5013, 0.5067, 0.4923, 0.3264]
            std = [0.1948, 0.1989, 0.2117, 0.199]
        elif args.dataset == 'rgbtvg_mfad':
            mean = [0.4733, 0.4695, 0.4622, 0.3393]
            std = [0.1654, 0.1646, 0.1749, 0.2063]
        elif args.dataset == 'rgbtvg_mixup':
            mean = [0.5103, 0.5111, 0.502, 0.3735]
            std = [0.1926, 0.1973, 0.2091, 0.2289]
        else:
            mean = [0.5, 0.5, 0.5, 0.5]
            std = [0.2, 0.2, 0.2, 0.2]
    elif args.modality == 'rgb':
        if args.dataset == 'rgbtvg_flir':
            mean = [0.631, 0.6401, 0.632]
            std = [0.2152, 0.227, 0.2439]
        elif args.dataset == 'rgbtvg_m3fd':
            mean = [0.5013, 0.5067, 0.4923]
            std = [0.1948, 0.1989, 0.2117]
        elif args.dataset == 'rgbtvg_mfad':
            mean = [0.4733, 0.4695, 0.4622]
            std = [0.1654, 0.1646, 0.1749]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
    elif args.modality == 'ir':
        if args.dataset == 'rgbtvg_flir':
            mean = [0.5337, 0.5337, 0.5337]
            std = [0.2562, 0.2562, 0.2562]
        elif args.dataset == 'rgbtvg_m3fd':
            mean = [0.3264, 0.3264, 0.3264]
            std = [0.199, 0.199, 0.199]
        elif args.dataset == 'rgbtvg_mfad':
            mean = [0.3393, 0.3393, 0.3393]
            std = [0.2063, 0.2063, 0.2063]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.2, 0.2, 0.2]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    return np.array(mean), np.array(std)


def denormalize_image(img_tensor, mean, std, modality='rgb'):
    """反归一化图像张量"""
    img = img_tensor.cpu().numpy()
    if len(img.shape) == 4:  # batch dimension
        img = img[0]
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    
    # 反归一化
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 如果是RGBT，只取前3个通道用于可视化
    if img.shape[2] == 4:
        img = img[:, :, :3]
    
    # 处理IR图像：转换为灰度显示
    if modality == 'ir' and img.shape[2] == 3:
        # 取第一个通道作为灰度图
        gray_img = np.ascontiguousarray(img[:, :, 0])
        # 为了绘制彩色bbox，转换回3通道
        img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    
    return img


def visualize_oneref_result(args, img_tensor, text, pred_bbox, gt_bbox, 
                            pred_mask=None, gt_mask=None, 
                            sample_idx=0, output_dir=None):
    """
    可视化OneRef模型的预测结果
    
    Args:
        args: 参数对象
        img_tensor: 图像张量 [C, H, W] 或 [B, C, H, W]
        text: 文本描述
        pred_bbox: 预测边界框 [x, y, w, h] (归一化)
        gt_bbox: 真实边界框 [x, y, w, h] (归一化)
        pred_mask: 预测分割掩码 [H, W] (可选)
        gt_mask: 真实分割掩码 [H, W] (可选)
        sample_idx: 样本索引
        output_dir: 输出目录
    """
    # 获取归一化参数
    mean, std = get_normalization_params(args)
    
    # 反归一化图像
    img = denormalize_image(img_tensor, mean, std, args.modality)
    img_h, img_w = img.shape[:2]
    
    # 创建可视化图像
    vis_img = img.copy()
    
    # 转换边界框格式
    imsize = args.imsize
    pred_bbox_xyxy = xywh2xyxy(pred_bbox.unsqueeze(0) if isinstance(pred_bbox, torch.Tensor) else torch.tensor(pred_bbox).unsqueeze(0))
    gt_bbox_xyxy = xywh2xyxy(gt_bbox.unsqueeze(0) if isinstance(gt_bbox, torch.Tensor) else torch.tensor(gt_bbox).unsqueeze(0))
    
    pred_coords = (imsize * pred_bbox_xyxy[0].cpu().numpy()).astype(int)
    gt_coords = (imsize * gt_bbox_xyxy[0].cpu().numpy()).astype(int)
    
    pred_x_min, pred_y_min, pred_x_max, pred_y_max = pred_coords
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_coords
    
    # 确保坐标在图像范围内
    pred_x_min = max(0, min(pred_x_min, img_w - 1))
    pred_y_min = max(0, min(pred_y_min, img_h - 1))
    pred_x_max = max(0, min(pred_x_max, img_w - 1))
    pred_y_max = max(0, min(pred_y_max, img_h - 1))
    
    gt_x_min = max(0, min(gt_x_min, img_w - 1))
    gt_y_min = max(0, min(gt_y_min, img_h - 1))
    gt_x_max = max(0, min(gt_x_max, img_w - 1))
    gt_y_max = max(0, min(gt_y_max, img_h - 1))
    
    # 绘制分割掩码（如果提供）
    if pred_mask is not None:
        if isinstance(pred_mask, torch.Tensor):
            pred_mask_np = pred_mask.cpu().numpy()
        else:
            pred_mask_np = pred_mask
        
        # 处理掩码维度：可能是 [H, W] 或 [1, H, W] 或 [B, H, W]
        if len(pred_mask_np.shape) == 3:
            if pred_mask_np.shape[0] == 1:
                pred_mask_np = pred_mask_np[0]
            else:
                pred_mask_np = pred_mask_np[0]  # 取第一个
        
        # 调整掩码大小到图像大小
        if pred_mask_np.shape != (img_h, img_w):
            pred_mask_np = cv2.resize(pred_mask_np.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # 归一化掩码到 [0, 1]
        if pred_mask_np.max() > 1.0:
            pred_mask_np = pred_mask_np / 255.0
        
        # 创建彩色掩码覆盖层
        mask_overlay = np.zeros_like(vis_img)
        mask_overlay[pred_mask_np > 0.5] = [0, 255, 0]  # 绿色
        vis_img = cv2.addWeighted(vis_img, 0.7, mask_overlay, 0.3, 0)
    
    if gt_mask is not None:
        if isinstance(gt_mask, torch.Tensor):
            gt_mask_np = gt_mask.cpu().numpy()
        else:
            gt_mask_np = gt_mask
        
        # 处理掩码维度：可能是 [H, W] 或 [1, H, W] 或 [B, H, W]
        if len(gt_mask_np.shape) == 3:
            if gt_mask_np.shape[0] == 1:
                gt_mask_np = gt_mask_np[0]
            else:
                gt_mask_np = gt_mask_np[0]  # 取第一个
        
        # 调整掩码大小到图像大小
        if gt_mask_np.shape != (img_h, img_w):
            gt_mask_np = cv2.resize(gt_mask_np.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # 归一化掩码到 [0, 1]
        if gt_mask_np.max() > 1.0:
            gt_mask_np = gt_mask_np / 255.0
        
        # 创建真实掩码轮廓
        mask_uint8 = (gt_mask_np > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 2)  # 红色轮廓
    
    # 绘制预测边界框（绿色）
    cv2.rectangle(vis_img, (pred_x_min, pred_y_min), (pred_x_max, pred_y_max), (0, 255, 0), 2)
    
    # 绘制真实边界框（红色）
    cv2.rectangle(vis_img, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (255, 0, 0), 2)
    
    # 添加文本标签
    text_y = 30
    cv2.putText(vis_img, f"Text: {text[:50]}", (10, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Text: {text[:50]}", (10, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # 添加图例
    legend_y = img_h - 80
    cv2.rectangle(vis_img, (10, legend_y), (30, legend_y + 20), (0, 255, 0), -1)  # 预测框
    cv2.putText(vis_img, "Pred", (35, legend_y + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.rectangle(vis_img, (10, legend_y + 25), (30, legend_y + 45), (255, 0, 0), -1)  # 真实框
    cv2.putText(vis_img, "GT", (35, legend_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 保存图像
    if output_dir is None:
        output_dir = getattr(args, 'visual_output_dir', './visual_result/oneref')
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"oneref_result_{sample_idx:06d}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    return vis_img, output_path


def visualize_oneref_batch(args, img_tensors, texts, pred_bboxes, gt_bboxes,
                          pred_masks=None, gt_masks=None,
                          start_idx=0, output_dir=None):
    """
    批量可视化OneRef模型的预测结果
    
    Args:
        args: 参数对象
        img_tensors: 图像张量 [B, C, H, W]
        texts: 文本描述列表
        pred_bboxes: 预测边界框 [B, 4]
        gt_bboxes: 真实边界框 [B, 4]
        pred_masks: 预测分割掩码 [B, H, W] (可选)
        gt_masks: 真实分割掩码 [B, H, W] (可选)
        start_idx: 起始索引
        output_dir: 输出目录
    """
    batch_size = img_tensors.tensors.size(0) if hasattr(img_tensors, 'tensors') else img_tensors.size(0)
    output_paths = []
    
    for i in range(batch_size):
        # 提取单个样本
        if hasattr(img_tensors, 'tensors'):
            img_tensor = img_tensors.tensors[i]
        else:
            img_tensor = img_tensors[i]
        
        text = texts[i] if isinstance(texts, list) else texts
        pred_bbox = pred_bboxes[i] if len(pred_bboxes.shape) > 1 else pred_bboxes
        gt_bbox = gt_bboxes[i] if len(gt_bboxes.shape) > 1 else gt_bboxes
        
        pred_mask = pred_masks[i] if pred_masks is not None and len(pred_masks.shape) > 2 else None
        gt_mask = gt_masks[i] if gt_masks is not None and len(gt_masks.shape) > 2 else None
        
        _, output_path = visualize_oneref_result(
            args, img_tensor, text, pred_bbox, gt_bbox,
            pred_mask, gt_mask,
            sample_idx=start_idx + i,
            output_dir=output_dir
        )
        output_paths.append(output_path)
    
    return output_paths


def visualize_oneref_pred_only(args, img_tensor, text, pred_bbox, sample_idx=0, output_dir=None):
    """
    仅可视化预测框（无GT框）
    """
    mean, std = get_normalization_params(args)
    img = denormalize_image(img_tensor, mean, std, args.modality)
    img_h, img_w = img.shape[:2]

    vis_img = img.copy()

    imsize = args.imsize
    if isinstance(pred_bbox, torch.Tensor):
        pred_bbox_xyxy = xywh2xyxy(pred_bbox.unsqueeze(0))
        pred_coords = (imsize * pred_bbox_xyxy[0].cpu().numpy()).astype(int)
    else:
        pred_bbox_xyxy = xywh2xyxy(torch.tensor(pred_bbox).unsqueeze(0))
        pred_coords = (imsize * pred_bbox_xyxy[0].cpu().numpy()).astype(int)

    x_min, y_min, x_max, y_max = pred_coords
    x_min = max(0, min(x_min, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    x_max = max(0, min(x_max, img_w - 1))
    y_max = max(0, min(y_max, img_h - 1))

    # 绘制预测框（绿色），不添加文本
    cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    if output_dir is None:
        output_dir = getattr(args, 'visual_output_dir', './visual_result/oneref')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存图片（不添加文本）
    output_path = os.path.join(output_dir, f"oneref_pred_{sample_idx:06d}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    # 保存文本到txt文件
    txt_path = os.path.join(output_dir, f"oneref_pred_{sample_idx:06d}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return vis_img, output_path
