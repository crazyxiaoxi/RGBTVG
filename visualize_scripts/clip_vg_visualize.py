#!/usr/bin/env python3
"""
CLIP_VG模型可视化脚本
用于生成CLIP_VG模型在数据集上的预测结果可视化
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import build_model
from datasets import make_transforms
from utils.misc import NestedTensor
from models.clip import clip


def get_args_parser():
    parser = argparse.ArgumentParser('CLIP_VG Visualization', add_help=False)
    
    # 基本参数
    parser.add_argument('--model_checkpoint', required=True, type=str, help='模型checkpoint路径')
    parser.add_argument('--label_file', required=True, type=str, help='数据标注文件路径')
    parser.add_argument('--dataroot', required=True, type=str, help='图像数据根目录')
    parser.add_argument('--output_dir', default='./visual_result/clip_vg', type=str, help='输出目录')
    
    # 数据集参数
    parser.add_argument('--dataset', default='rgbtvg_flir', type=str, help='数据集名称')
    parser.add_argument('--modality', default='rgb', type=str, choices=['rgb', 'ir', 'rgbt'], help='图像模态')
    parser.add_argument('--num_samples', default=100, type=int, help='可视化样本数量')
    parser.add_argument('--start_idx', default=0, type=int, help='起始索引')
    
    # 训练相关参数（模型初始化需要，但可视化时不使用）
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_bert', default=0., type=float)
    parser.add_argument('--lr_visu_cnn', default=0., type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_power', default=0.9, type=float)
    parser.add_argument('--clip_max_norm', default=0., type=float)
    parser.add_argument('--optimizer', default='rmsprop', type=str)
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--sup_type', default='full', type=str)
    parser.add_argument('--old_dataloader', default=True, type=bool)
    
    # Augmentation options (推理时不使用)
    parser.add_argument('--aug_blur', action='store_true')
    parser.add_argument('--aug_crop', action='store_true')
    parser.add_argument('--aug_scale', action='store_true')
    parser.add_argument('--aug_translate', action='store_true')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='CLIP_VG', help='模型名称')
    parser.add_argument('--model', type=str, default='ViT-B/16', help='CLIP模型类型')
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    
    # 图像参数
    parser.add_argument('--imsize', default=224, type=int, help='图像尺寸')
    parser.add_argument('--emb_size', default=512, type=int, help='embedding维度')
    
    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float)
    parser.add_argument('--vl_nheads', default=8, type=int)
    parser.add_argument('--vl_hidden_dim', default=512, type=int)
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int)
    parser.add_argument('--vl_enc_layers', default=6, type=int)
    
    # 其他参数
    parser.add_argument('--max_query_len', default=77, type=int)
    parser.add_argument('--prompt', type=str, default='{pseudo_query}', help='Prompt template')
    parser.add_argument('--light', dest='light', default=False, action='store_true')
    parser.add_argument('--gpu_id', default='0', type=str, help='GPU ID')
    
    # Prompt Engineering & Advanced features
    parser.add_argument('--use_cot_prompt', action='store_true')
    parser.add_argument('--cot_length', type=int, default=0)
    parser.add_argument('--use_contrastive_loss', action='store_true')
    parser.add_argument('--use_rtcc_constrain_loss', action='store_true')
    parser.add_argument('--use_mask_loss', action='store_true')
    parser.add_argument('--use_seg_mask', action='store_true')
    parser.add_argument('--retrain', default='', help='retrain from checkpoint')
    parser.add_argument('--adapt_mlp', action='store_true')
    parser.add_argument('--save_hilora_clip', action='store_true')
    parser.add_argument('--hi_lora_stage', default=0, type=int)
    parser.add_argument('--hi_lora_retrain', default='')
    parser.add_argument('--hi_lora_clip', default='', type=str)
    parser.add_argument('--mixup_pretrain', action='store_true')
    parser.add_argument('--enable_adaptive_weights', action='store_true')
    
    # Additional dataset parameters
    parser.add_argument('--data_root', type=str, default='./ln_data/')
    parser.add_argument('--split_root', type=str, default='data')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=13, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true')
    parser.add_argument('--eval_set', default='test', type=str)
    parser.add_argument('--eval_model', default='', type=str)
    parser.add_argument('--normalize_before', action='store_true', help='If true, use normalize_before')
    parser.add_argument('--distributed', default=False, type=bool, help='distributed training')
    
    return parser


def load_model(args, device):
    """加载CLIP_VG模型"""
    print(f"Loading CLIP_VG model from: {args.model_checkpoint}")
    
    # 加载checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    # 如果checkpoint中有保存的args，使用checkpoint的配置来构建模型
    if 'args' in checkpoint:
        print("Using model configuration from checkpoint...")
        model_args = checkpoint['args']
        # 保留可视化相关的参数
        model_args.gpu_id = args.gpu_id
        model_args.output_dir = args.output_dir
        model_args.num_samples = args.num_samples
        model_args.start_idx = args.start_idx
        model_args.label_file = args.label_file
        model_args.dataroot = args.dataroot
        # 确保eval_model参数存在
        if not hasattr(model_args, 'eval_model'):
            model_args.eval_model = getattr(args, 'eval_model', '')
        args = model_args  # 使用checkpoint中的配置
    
    # 确保必要的参数存在
    if not hasattr(args, 'eval_model'):
        args.eval_model = ''
    
    # 构建模型 (CLIP_VG的build_model只返回模型对象，不是元组)
    model = build_model(args)
    model.to(device)
    
    # 加载模型权重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print("Model loaded successfully!")
    return model, args  # 返回更新后的args


def load_dataset(label_file):
    """加载数据集标注文件"""
    print(f"Loading dataset from: {label_file}")
    data = torch.load(label_file)
    print(f"Total samples in dataset: {len(data)}")
    return data


def process_image(args, img_path, text, transform):
    """处理单张图像，返回transform后的图像用于模型推理和原始图像用于可视化"""
    pil_img_original = None  # 保存原始图像用于可视化
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
            return None, None, None
        
        # 加载RGB和IR图像，模仿数据加载器的逻辑
        import numpy as np
        rgb_img = Image.open(rgb_path).convert('RGB')
        ir_img = Image.open(ir_path)
        
        # 保存原始RGB图像用于可视化
        pil_img_original = rgb_img.copy()
        
        # 将RGB和IR合并为4通道图像（与数据加载器一致）
        np_rgb = np.array(rgb_img)
        np_ir = np.array(ir_img)
        if np_ir.ndim == 3:
            np_ir = np_ir[..., 0]
        np_ir = np.expand_dims(np_ir, axis=-1)
        np_combined = np.concatenate([np_rgb, np_ir], axis=-1)
        rgbt_img = Image.fromarray(np_combined)
        
        # 获取图像尺寸
        w, h = rgb_img.size
        full_box_xyxy = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], dtype=torch.float32)
        
        # 使用transform处理RGBT图像，确保使用正确的归一化参数
        input_dict = {'img': rgbt_img, 'box': full_box_xyxy, 'text': text}
        input_dict = transform(input_dict)
        
        return input_dict['img'], input_dict['mask'], pil_img_original
        
    elif args.modality == 'ir':
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
    else:  # RGB
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return None, None, None
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


def save_visualization(args, pil_img_original, text, pred_bbox, sample_idx, output_dir):
    """保存单个可视化结果（使用原始图像而不是transform后的图像）"""
    # 直接使用原始图像，不需要反归一化
    img_np = np.array(pil_img_original)
    
    # 转换bbox到像素坐标
    h, w = img_np.shape[:2]
    if isinstance(pred_bbox, torch.Tensor):
        pred_bbox = pred_bbox.cpu().numpy()
    
    # 假设pred_bbox是xywh格式，转换为xyxy
    x_center, y_center, bbox_w, bbox_h = pred_bbox
    x_min = int((x_center - bbox_w / 2) * w)
    y_min = int((y_center - bbox_h / 2) * h)
    x_max = int((x_center + bbox_w / 2) * w)
    y_max = int((y_center + bbox_h / 2) * h)
    
    # 限制在图像范围内
    x_min = max(0, min(x_min, w - 1))
    y_min = max(0, min(y_min, h - 1))
    x_max = max(0, min(x_max, w - 1))
    y_max = max(0, min(y_max, h - 1))
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 直接在原始图像上绘制bbox（所有模态都使用RGB图像）
    vis_img = np.ascontiguousarray(img_np)
    cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    output_path = os.path.join(output_dir, f"clip_vg_pred_{sample_idx:06d}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    # 保存文本到txt文件
    txt_path = os.path.join(output_dir, f"clip_vg_pred_{sample_idx:06d}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return output_path


def visualize_dataset(args):
    """可视化数据集"""
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型（可能会更新args为checkpoint中的配置）
    model, args = load_model(args, device)
    
    # CLIP_VG使用CLIP的内置tokenizer，不需要单独初始化
    print("Using CLIP's built-in tokenizer")
    
    # 加载数据集
    dataset = load_dataset(args.label_file)
    
    # 选择要可视化的样本
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    samples_to_process = dataset[args.start_idx:end_idx]
    print(f"Visualizing samples {args.start_idx} to {end_idx - 1} (total: {len(samples_to_process)})")
    
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
            result = process_image(args, img_path, text, transform)
            if result is None or len(result) != 3:
                fail_count += 1
                continue
            img_tensor, img_mask, pil_img_original = result
            if img_tensor is None:
                fail_count += 1
                continue
            
            # 准备模型输入
            img_tensor = img_tensor.unsqueeze(0).to(device)
            img_mask = img_mask.unsqueeze(0).to(device)
            img_nt = NestedTensor(img_tensor, img_mask)
            
            # CLIP_VG使用CLIP的tokenizer处理文本
            text_tokens = clip.tokenize([text]).to(device)  # (1, 77)
            text_nt = NestedTensor(text_tokens, torch.zeros_like(text_tokens, dtype=torch.bool))
            
            # 模型推理
            with torch.no_grad():
                # CLIP_VG模型期望(img_nt, text_nt)
                pred_boxes = model(img_nt, text_nt)
            
            # # 调试信息（只显示前5个样本）
            # if idx < 5:
            #     print(f"\n样本{sample_idx}: {img_filename}")
            #     print(f"  文本: {text[:50]}...")
            #     print(f"  预测bbox: {pred_boxes[0].cpu().numpy()}")
            #     print(f"  GT bbox: {bbox_gt}")
                
            #     # 计算与前一个的差异
            #     if idx > 0 and 'last_pred' in locals():
            #         diff = abs(pred_boxes[0].cpu().numpy() - last_pred).sum()
            #         print(f"  与上一个预测差异: {diff}")
            #     last_pred = pred_boxes[0].cpu().numpy()
            
            # 可视化结果（使用原始图像）
            bbox = pred_boxes[0].cpu()
            
            out_path = save_visualization(
                args, pil_img_original, text, bbox,
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
    parser = argparse.ArgumentParser('CLIP_VG Visualization', parents=[get_args_parser()])
    args = parser.parse_args()
    
    visualize_dataset(args)


if __name__ == '__main__':
    main()
