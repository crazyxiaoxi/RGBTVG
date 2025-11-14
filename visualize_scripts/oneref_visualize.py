#!/usr/bin/env python3
"""
OneRef模型可视化脚本
基于数据集文件（.pth）进行批量可视化预测结果
"""
import os
import sys
import torch
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# 添加父目录到path以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# OneRef model imports
from timm.models import create_model
import models.utils as beit3_utils
import models.OneRef_model as OneRef_model
import models.modeling_vqkd as modeling_vqkd

# Utils
from datasets import make_transforms
from utils.misc import NestedTensor
from utils.box_utils import xywh2xyxy

# 导入公共可视化工具
from utils_visualization import process_image, save_pred_visualization, load_dataset, save_combined_pred_visualization, generate_prediction_statistics


def get_args_parser():
    parser = argparse.ArgumentParser('OneRef Visualization Script', add_help=False)
    
    # Basic model parameters
    parser.add_argument('--model_name', type=str, default='OneRef', help='model name')
    parser.add_argument('--old_dataloader', default=False, type=bool)
    parser.add_argument('--sup_type', default='full', type=str)
    
    # Model parameters
    parser.add_argument('--model', default='beit3_large_patch16_224', type=str,
                        help='Model architecture')
    parser.add_argument('--task', type=str, default='grounding',
                        help='Task type')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--sentencepiece_model', type=str,
                        default='../dataset_and_pretrain_model/pretrain_model/pretrained_weights/BEIT3/beit3.spm',
                        help='Sentencepiece model path')
    
    # Data parameters
    parser.add_argument('--label_file', type=str, required=True,
                        help='Path to label .pth file (e.g., rgbtvg_flir_train.pth)')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Root directory for images')
    parser.add_argument('--dataset', default='rgbtvg_flir', type=str,
                        help='Dataset name (e.g., rgbtvg_flir, rgbtvg_m3fd)')
    parser.add_argument('--modality', default='rgb', type=str,
                        choices=['rgb', 'ir', 'rgbt'],
                        help='Modality type')
    
    # Visualization parameters
    parser.add_argument('--output_dir', type=str, default='./visual_result/oneref',
                        help='Output directory for visualization results')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to visualize (0 means all)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting index in the dataset')
    
    # Model training parameters (needed for model initialization)
    parser.add_argument('--enable_seg_mask', action='store_true', help="If true, use segmentation mask, otherwise use box mask.")
    parser.add_argument('--frozen_backbone', action='store_true', help="If true, frozen BEiT-3", default=False)
    parser.add_argument('--use_contrastive_loss', action='store_true', help="If true, use contrastive loss")
    parser.add_argument('--use_box_mask_constraints', action='store_true', help="If true, use box mask constraints")
    parser.add_argument('--use_mask_loss', action='store_true', help="If true, use segmentation loss")
    parser.add_argument('--enable_ref_mlm', action='store_true', help="If true, use mlm loss")
    parser.add_argument('--enable_ref_mim', action='store_true', help="If true, use mim loss")
    parser.add_argument('--enable_dynamic_mim', action='store_true', help="If true, use dynamic mim loss")
    parser.add_argument('--mim_mask_ratio', type=float, default=0.4)
    parser.add_argument('--text_mask_prob', type=float, default=0.4)
    parser.add_argument('--drop_worst_ratio', type=float, default=0.2)
    parser.add_argument('--drop_worst_after', type=int, default=12000)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--checkpoint_activations', action='store_true', default=None)
    
    # MIM pretraining parameters
    parser.add_argument('--early_layers', default=21, type=int, help='early_layers, default 9 for base and 21 for large')
    parser.add_argument('--head_layers', default=2, type=int, help='head_layers')
    parser.add_argument('--mim_mid_layer', default=0, type=int, help='mim_mid_layer,set 0 or 9')
    parser.add_argument('--shared_lm_head', default=True, type=lambda x: x.lower() == 'true', help='shared lm head')
    parser.add_argument('--num_mask_patches', default=75, type=int, help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
    
    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=32, type=int, help='hidden dimension of codebook')
    parser.add_argument("--tokenizer_weight", type=str, default="../dataset_and_pretrain_model/pretrain_model/pretrained_weights/vqkd/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth")
    parser.add_argument("--tokenizer_model", type=str, default="vqkd_encoder_base_decoder_3x768x12_clip")
    
    # Model settings
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--vocab_size', type=int, default=64010)
    parser.add_argument('--num_max_bpe_tokens', type=int, default=64)
    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--use_regress_box', action='store_true', default=True)
    
    # Transformer parameters
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int, help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--emb_size', default=512, type=int, help='fusion module embedding dimensions')
    
    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=512, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    parser.add_argument('--vl_dec_layers', default=6, type=int,
                        help='Number of decoders in the vision-language transformer')
    
    # Text parameters
    parser.add_argument('--max_query_len', default=77, type=int, help='maximum time steps (lang length) per batch')
    parser.add_argument('--prompt', type=str, default='{pseudo_query}', help="Prompt template")
    
    # Other parameters
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    
    # Device
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='GPU ID to use')
    
    return parser




# ==================== 模型加载和处理函数 ====================

def load_model(args):
    """加载OneRef模型"""
    print(f"Loading model from: {args.model_checkpoint}")
    
    # Generate model config
    if not args.model.endswith(args.task):
        if args.task == "grounding":
            model_config = "%s_grounding" % args.model
        else:
            model_config = "%s_%s" % (args.model, args.task)
    else:
        model_config = args.model
    
    print(f"Model config: {model_config}")
    
    # Create model
    model = create_model(
        model_config,
        sys_args=args,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations if args.checkpoint_activations is not None else False,
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    print(f"Model loaded from epoch: {checkpoint.get('epoch', 'N/A')}")
    
    model.to(args.device)
    model.eval()
    
    return model






def visualize_dataset(args):
    """可视化数据集，按图片分组处理"""
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
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.visual_output_dir = args.output_dir
    
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
                texts = [text]
                
                # 模型推理
                with torch.no_grad():
                    pred_box, pred_seg, img_cls, text_cls = model(img_nt.tensors, img_nt.mask, texts)
                bbox = pred_box[0].cpu()
                
                predictions.append({
                    'bbox': bbox,
                    'text': text,
                    'sample_idx': item['sample_idx']
                })
            
            # 保存合并的预测可视化
            save_combined_pred_visualization(
                args, pil_img_original, pil_img_ir, predictions, 
                img_filename, args.output_dir, "oneref"
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
    generate_prediction_statistics(args.output_dir, prediction_stats, args.dataset, args.modality, "oneref")
    
    print(f"\nVisualization complete!")
    print(f"Total images processed: {processed_images}")
    print(f"Total predictions: {len(samples_to_process)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Statistics report saved to: {args.output_dir}/prediction_statistics.txt")


def main():
    parser = argparse.ArgumentParser('OneRef Visualization', parents=[get_args_parser()])
    args = parser.parse_args()
    
    visualize_dataset(args)


if __name__ == '__main__':
    main()
