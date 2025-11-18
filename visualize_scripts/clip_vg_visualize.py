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

from utils_visualization import process_image, save_pred_visualization, load_dataset, generate_prediction_statistics

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
    parser.add_argument('--num_samples', default=0, type=int, help='可视化样本数量（0表示使用整个数据集）')
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
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print("Model loaded successfully!")
    return model, args  # 返回更新后的args


# load_dataset函数已移至utils_visualization.py


# process_image函数已移至utils_visualization.py


# save_visualization函数已移至utils_visualization.py


def visualize_dataset(args):
    """可视化数据集，按图片分组处理"""
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, args = load_model(args, device)
    
    # 加载数据集
    dataset = load_dataset(args.label_file)
    
    # 确定要可视化的样本范围
    end_idx = args.start_idx + args.num_samples if args.num_samples > 0 else len(dataset)
    end_idx = min(end_idx, len(dataset))
    samples_to_process = dataset[args.start_idx:end_idx]
    
    print(f"Visualizing samples {args.start_idx} to {end_idx-1} (total: {len(samples_to_process)})")
    
    # 按图片文件名分组
    image_groups = {}
    for idx, item in enumerate(samples_to_process):
        sample_idx = args.start_idx + idx
        
        # 解析数据格式
        if str(args.dataset).startswith('rgbtvg'):
            img_filename = item[0]
            img_size = item[1]
            bbox_gt = item[2]
            text = item[3]
            lighting = item[4] if len(item) > 4 else None
            scale_cls = item[5] if len(item) > 5 else None
        else:
            img_filename = item[0]
            bbox_gt = item[2]
            text = item[3]
        
        # 按图片文件名分组
        if img_filename not in image_groups:
            image_groups[img_filename] = []
        
        image_groups[img_filename].append({
            'sample_idx': sample_idx,
            'bbox_gt': bbox_gt,
            'text': text,
            'item': item
        })
    
    print(f"Found {len(image_groups)} unique images with annotations")
    
    # 构建变换
    transform = make_transforms(args, 'val')
    
    # 处理每个图片组
    success_count = 0
    fail_count = 0
    processed_images = 0
    prediction_stats = []  # 用于统计每个图片的预测数量
    
    for img_filename, group_items in image_groups.items():
        processed_images += 1
        img_path = os.path.join(args.dataroot, img_filename)
        
        try:
            # 使用第一个样本来处理图像（所有样本使用同一张图）
            first_item = group_items[0]
            result = process_image(args, img_path, first_item['text'], transform)
            if result is None:
                fail_count += len(group_items)
                continue
            
            # 根据模态解析返回值
            if args.modality == 'rgbt':
                if len(result) != 4:
                    fail_count += len(group_items)
                    continue
                img_tensor, img_mask, pil_img_original, pil_img_ir = result
            else:
                if len(result) != 3:
                    fail_count += len(group_items)
                    continue
                img_tensor, img_mask, pil_img_original = result
                pil_img_ir = None
            
            if img_tensor is None:
                fail_count += len(group_items)
                continue
            
            # 为每个查询进行预测
            predictions = []
            for item in group_items:
                text = item['text']
                
                # 准备模型输入
                img_tensor_batch = img_tensor.unsqueeze(0).to(device)
                img_mask_batch = img_mask.unsqueeze(0).to(device)
                img_nt = NestedTensor(img_tensor_batch, img_mask_batch)
                
                # 文本处理
                tokenizer = clip.tokenize([text], truncate=True).to(device)
                word_id_tensor = tokenizer
                word_mask_tensor = (word_id_tensor != 0).float()
                text_nt = NestedTensor(word_id_tensor, word_mask_tensor)
                
                # 模型推理
                with torch.no_grad():
                    pred_boxes = model(img_nt, text_nt)
                bbox = pred_boxes[0].cpu()
                
                predictions.append({
                    'bbox': bbox,
                    'text': text,
                    'sample_idx': item['sample_idx']
                })
            
            # 保存合并的预测可视化（单图，多框，编号+颜色区分）
            save_pred_visualization(
                args, pil_img_original, pil_img_ir, predictions,
                img_filename, args.output_dir, "clip_vg"
            )
            
            # 记录统计信息
            prediction_stats.append({
                'image': img_filename,
                'predictions': len(predictions)
            })
            
            success_count += len(group_items)
            print(f"Processed image {processed_images}/{len(image_groups)}: {img_filename} ({len(group_items)} predictions)")
        
        except Exception as e:
            print(f"Error processing image {img_filename}: {str(e)}")
            fail_count += len(group_items)
            continue
    
    # 生成统计报告
    generate_prediction_statistics(args.output_dir, prediction_stats, args.dataset, args.modality, "clip_vg")
    
    print(f"\nVisualization complete!")
    print(f"Total images processed: {processed_images}")
    print(f"Total predictions: {len(samples_to_process)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Statistics report saved to: {args.output_dir}/prediction_statistics.txt")


def main():
    parser = argparse.ArgumentParser('CLIP_VG Visualization', parents=[get_args_parser()])
    args = parser.parse_args()
    
    visualize_dataset(args)


if __name__ == '__main__':
    main()
