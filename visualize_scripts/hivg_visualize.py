#!/usr/bin/env python3
"""
HiVG模型可视化脚本
基于数据集文件（.pth）进行批量可视化预测结果
"""
import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# 添加父目录到path以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# HiVG model imports
from models import build_model
from datasets import make_transforms
from utils.misc import NestedTensor
from utils.visual_utils import visualization
import cv2
from utils.box_utils import xywh2xyxy


def get_args_parser():
    parser = argparse.ArgumentParser('HiVG Visualization Script', add_help=False)
    
    # Basic model parameters
    parser.add_argument('--model_name', type=str, default='HiVG', help='model name')
    parser.add_argument('--sup_type', default='full', type=str)
    
    # Model parameters
    parser.add_argument('--model', type=str, default='ViT-B/16', 
                        help="Name of CLIP model (ViT-B/16 or ViT-L/14)")
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    
    # Data parameters
    parser.add_argument('--label_file', type=str, required=True,
                        help='Path to label .pth file')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Root directory for images')
    parser.add_argument('--dataset', default='rgbtvg_flir', type=str,
                        help='Dataset name')
    parser.add_argument('--modality', default='rgb', type=str,
                        choices=['rgb', 'ir', 'rgbt'],
                        help='Modality type')
    
    # Visualization parameters
    parser.add_argument('--output_dir', type=str, default='./visual_result/hivg',
                        help='Output directory for visualization results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to visualize (0 means all)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index in the dataset')
    
    # Model architecture parameters
    parser.add_argument('--extract_layer', default=0, type=int)
    parser.add_argument('--warmup', action='store_true', help="If true, vision adapt layer is null")
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--imsize', default=224, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int)
    
    # Vision-Language Transformer
    parser.add_argument('--use_vl_type_embed', action='store_true')
    parser.add_argument('--vl_dropout', default=0.0, type=float)
    parser.add_argument('--vl_nheads', default=8, type=int)
    parser.add_argument('--vl_hidden_dim', default=512, type=int)
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int)
    parser.add_argument('--vl_enc_layers', default=6, type=int)
    parser.add_argument('--vl_dec_layers', default=6, type=int)
    
    # Text parameters
    parser.add_argument('--max_query_len', default=77, type=int)
    parser.add_argument('--prompt', type=str, default='{pseudo_query}')
    
    # Training parameters (needed for model initialization)
    parser.add_argument('--adapt_mlp', action='store_true')
    parser.add_argument('--use_loss_coef', action='store_true')
    parser.add_argument('--normalize_before', action='store_true')
    parser.add_argument('--hi_lora_stage', default=0, type=int, help='lora stage')
    parser.add_argument('--use_seg_mask', action='store_true')
    parser.add_argument('--use_mask_loss', action='store_true')
    parser.add_argument('--mixup_pretrain', action='store_true')
    parser.add_argument('--enable_adaptive_weights', action='store_true')
    
    # Dummy parameters for compatibility
    parser.add_argument('--detr_model', default='', type=str)
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str)
    parser.add_argument('--light', dest='light', default=False, action='store_true')
    
    # Device
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu_id', default='0', type=str)
    
    return parser


def load_model(args):
    """加载HiVG模型"""
    print(f"Loading model from: {args.model_checkpoint}")
    
    # 根据模型类型调整hidden_dim
    if args.model == "ViT-L/14" or args.model == "ViT-L/14@336px":
        args.vl_hidden_dim = 768
    
    # 构建模型
    model = build_model(args)
    
    # 加载checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys[:5]}...")  # 只打印前5个
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")
    
    print(f"Model loaded from epoch: {checkpoint.get('epoch', 'N/A')}")
    
    model.to(args.device)
    model.eval()
    
    return model


def load_dataset(label_file):
    """加载数据集标注文件"""
    print(f"Loading dataset from: {label_file}")
    data = torch.load(label_file)
    print(f"Total samples in dataset: {len(data)}")
    return data


def process_image(args, img_path, text, transform):
    """处理单张图像"""
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
            ir_path = None
        
        # 加载RGB图像
        if os.path.exists(rgb_path):
            img_rgb = Image.open(rgb_path).convert('RGB')
        else:
            print(f"Warning: RGB image not found: {rgb_path}")
            return None, None
        
        # 加载IR图像
        if ir_path and os.path.exists(ir_path):
            img_ir = Image.open(ir_path)
            np_rgb = np.array(img_rgb)
            np_ir = np.array(img_ir)
            
            if np_ir.ndim == 3:
                np_ir = np_ir[..., 0]
            
            np_ir = np.expand_dims(np_ir, axis=-1)
            np_combined = np.concatenate([np_rgb, np_ir], axis=-1)
            pil_img = Image.fromarray(np_combined)
        else:
            print(f"Warning: IR image not found: {ir_path}, using RGB only")
            pil_img = img_rgb
    
    elif args.modality == 'ir':
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return None, None
        # IR图像转换为RGB（3通道），与dataloader保持一致
        pil_img = Image.open(img_path).convert('RGB')
    
    else:  # RGB
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return None, None
        pil_img = Image.open(img_path).convert('RGB')
    
    # 获取图像尺寸
    w, h = pil_img.size
    full_box_xyxy = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], dtype=torch.float32)
    
    # 应用变换
    input_dict = {'img': pil_img, 'box': full_box_xyxy, 'text': text}
    input_dict = transform(input_dict)
    
    return input_dict['img'], input_dict['mask']


def save_visualization(args, img_tensor, text, pred_bbox, sample_idx, output_dir):
    """保存单个可视化结果（图片只显示bbox框，文本保存到txt文件）"""
    # 获取归一化参数
    if args.modality == 'rgbt':
        if args.dataset == 'rgbtvg_flir':
            mean, std = [0.631, 0.6401, 0.632, 0.5337], [0.2152, 0.227, 0.2439, 0.2562]
        elif args.dataset == 'rgbtvg_m3fd':
            mean, std = [0.5013, 0.5067, 0.4923, 0.3264], [0.1948, 0.1989, 0.2117, 0.199]
        elif args.dataset == 'rgbtvg_mfad':
            mean, std = [0.4733, 0.4695, 0.4622, 0.3393], [0.1654, 0.1646, 0.1749, 0.2063]
        elif args.dataset == 'rgbtvg_mixup':
            mean, std = [0.5103, 0.5111, 0.502, 0.3735], [0.1926, 0.1973, 0.2091, 0.2289]
        else:
            mean, std = [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.2, 0.2]
    elif args.modality == 'rgb':
        if args.dataset == 'rgbtvg_flir':
            mean, std = [0.631, 0.6401, 0.632], [0.2152, 0.227, 0.2439]
        elif args.dataset == 'rgbtvg_m3fd':
            mean, std = [0.5013, 0.5067, 0.4923], [0.1948, 0.1989, 0.2117]
        elif args.dataset == 'rgbtvg_mfad':
            mean, std = [0.4733, 0.4695, 0.4622], [0.1654, 0.1646, 0.1749]
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif args.modality == 'ir':
        if args.dataset == 'rgbtvg_flir':
            mean, std = [0.5337, 0.5337, 0.5337], [0.2562, 0.2562, 0.2562]
        elif args.dataset == 'rgbtvg_m3fd':
            mean, std = [0.3264, 0.3264, 0.3264], [0.199, 0.199, 0.199]
        elif args.dataset == 'rgbtvg_mfad':
            mean, std = [0.3393, 0.3393, 0.3393], [0.2063, 0.2063, 0.2063]
        else:
            mean, std = [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    mean = np.array(mean)
    std = np.array(std)
    
    # 反归一化图像
    img = img_tensor.cpu().numpy()
    img = img.transpose(1, 2, 0)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    img_h, img_w = img.shape[:2]
    
    # 转换边界框
    imsize = args.imsize
    pred_bbox_xyxy = xywh2xyxy(pred_bbox.unsqueeze(0))
    pred_coords = (imsize * pred_bbox_xyxy[0].cpu().numpy()).astype(int)
    
    x_min, y_min, x_max, y_max = pred_coords
    x_min = max(0, min(x_min, img_w - 1))
    y_min = max(0, min(y_min, img_h - 1))
    x_max = max(0, min(x_max, img_w - 1))
    y_max = max(0, min(y_max, img_h - 1))
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 处理不同模态的图像保存
    if args.modality == 'rgbt' and img.shape[2] == 4:
        # RGBT模态：保存两张图片（RGB彩色 + IR灰度）
        # 1. 保存RGB彩色图像
        rgb_img = np.ascontiguousarray(img[:, :, :3])
        cv2.rectangle(rgb_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        rgb_path = os.path.join(output_dir, f"hivg_pred_{sample_idx:06d}_rgb.jpg")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        # 2. 保存IR灰度图像
        ir_img = np.ascontiguousarray(img[:, :, 3])  # 第4个通道是IR
        ir_img_3ch = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)  # 转为3通道以绘制彩色bbox
        cv2.rectangle(ir_img_3ch, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        ir_path = os.path.join(output_dir, f"hivg_pred_{sample_idx:06d}_ir.jpg")
        cv2.imwrite(ir_path, cv2.cvtColor(ir_img_3ch, cv2.COLOR_RGB2BGR))
        
        output_path = rgb_path  # 返回RGB图像路径作为主要输出
        
    elif args.modality == 'ir':
        # IR图像：取第一个通道作为灰度图显示
        gray_img = np.ascontiguousarray(img[:, :, 0])
        vis_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        output_path = os.path.join(output_dir, f"hivg_pred_{sample_idx:06d}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
    else:  # RGB 或 RGBT但只显示RGB部分
        if img.shape[2] == 4:
            vis_img = np.ascontiguousarray(img[:, :, :3])  # 只取RGB部分
        else:
            vis_img = img.copy()
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        output_path = os.path.join(output_dir, f"hivg_pred_{sample_idx:06d}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    # 保存文本到txt文件
    txt_path = os.path.join(output_dir, f"hivg_pred_{sample_idx:06d}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return output_path


def visualize_dataset(args):
    """可视化数据集样本"""
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device(args.device)
    
    # 加载模型
    model = load_model(args)
    
    # 加载数据集
    dataset = load_dataset(args.label_file)
    
    # 确定要可视化的样本范围
    end_idx = args.start_idx + args.num_samples if args.num_samples > 0 else len(dataset)
    end_idx = min(end_idx, len(dataset))
    samples_to_process = dataset[args.start_idx:end_idx]
    
    print(f"Visualizing samples {args.start_idx} to {end_idx-1} (total: {len(samples_to_process)})")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 构建变换
    transform = make_transforms(args, 'val')
    
    # 处理每个样本
    success_count = 0
    fail_count = 0
    
    for idx, item in enumerate(samples_to_process):
        sample_idx = args.start_idx + idx
        
        # 解析数据格式: RGBTVG格式是 [img_file, img_size, bbox, phrase, lighting, scale_cls]
        if str(args.dataset).startswith('rgbtvg'):
            img_filename = item[0]
            img_size = item[1]  # 可能是字典 {"height": x, "width": y}
            bbox_gt = item[2]
            text = item[3]
            lighting = item[4] if len(item) > 4 else None
            scale_cls = item[5] if len(item) > 5 else None
        else:
            # 其他数据集格式: [img_file, _, bbox, phrase, ...]
            img_filename = item[0]
            bbox_gt = item[2]
            text = item[3]
        
        # 构建完整图像路径
        img_path = os.path.join(args.dataroot, img_filename)
        
        try:
            # 处理图像
            img_tensor, img_mask = process_image(args, img_path, text, transform)
            
            if img_tensor is None:
                fail_count += 1
                continue
            
            # 准备模型输入
            img_tensor = img_tensor.unsqueeze(0).to(device)
            img_mask = img_mask.unsqueeze(0).to(device)
            img_nt = NestedTensor(img_tensor, img_mask)
            texts = [text]
            
            # 模型推理
            with torch.no_grad():
                # HiVG模型返回tuple: (pred_box, logits_per_text, logits_per_image, visu_token_similarity, seg_mask)
                outputs = model(img_nt, texts)
                pred_boxes = outputs[0]  # pred_box是第一个元素
            
            # 可视化结果
            img_t = img_tensor[0].cpu()
            bbox = pred_boxes[0].cpu()
            
            out_path = save_visualization(
                args, img_t, text, bbox,
                sample_idx=sample_idx,
                output_dir=args.output_dir
            )
            
            success_count += 1
            if (idx + 1) % 10 == 0:
                print(f"Progress: {idx + 1}/{len(samples_to_process)} - Success: {success_count}, Failed: {fail_count}")
        
        except Exception as e:
            print(f"Error processing sample {sample_idx} ({img_filename}): {str(e)}")
            fail_count += 1
            continue
    
    print(f"\nVisualization complete!")
    print(f"Total processed: {len(samples_to_process)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser('HiVG Visualization', parents=[get_args_parser()])
    args = parser.parse_args()
    
    visualize_dataset(args)


if __name__ == '__main__':
    main()
