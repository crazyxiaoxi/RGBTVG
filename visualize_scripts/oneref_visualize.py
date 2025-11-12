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


# ==================== 可视化工具函数 ====================

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


def visualize_oneref_pred_only(args, img_tensor, text, pred_bbox, sample_idx=0, output_dir=None):
    """
    仅可视化预测框（无GT框）
    """
    mean, std = get_normalization_params(args)
    img = denormalize_image(img_tensor, mean, std, args.modality)
    img_h, img_w = img.shape[:2]

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

    if output_dir is None:
        output_dir = getattr(args, 'visual_output_dir', './visual_result/oneref')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 处理不同模态的图像保存
    if args.modality == 'rgbt' and img.shape[2] >= 4:
        # RGBT模态：保存两张图片（RGB彩色 + IR灰度）
        # 1. 保存RGB彩色图像
        rgb_img = np.ascontiguousarray(img[:, :, :3])
        cv2.rectangle(rgb_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        rgb_path = os.path.join(output_dir, f"oneref_pred_{sample_idx:06d}_rgb.jpg")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        
        # 2. 保存IR灰度图像
        ir_img = np.ascontiguousarray(img[:, :, 3])  # 第4个通道是IR
        ir_img_3ch = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2RGB)  # 转为3通道以绘制彩色bbox
        cv2.rectangle(ir_img_3ch, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        ir_path = os.path.join(output_dir, f"oneref_pred_{sample_idx:06d}_ir.jpg")
        cv2.imwrite(ir_path, cv2.cvtColor(ir_img_3ch, cv2.COLOR_RGB2BGR))
        
        output_path = rgb_path  # 返回RGB图像路径作为主要输出
        vis_img = rgb_img
        
    else:
        # RGB 或 IR 模态
        vis_img = img.copy()
        cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        output_path = os.path.join(output_dir, f"oneref_pred_{sample_idx:06d}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    
    # 保存文本到txt文件
    txt_path = os.path.join(output_dir, f"oneref_pred_{sample_idx:06d}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return vis_img, output_path


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
            # 无法确定路径结构，假设只有RGB
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
            
            # 如果IR图像是3通道，只取第一个通道
            if np_ir.ndim == 3:
                np_ir = np_ir[..., 0]
            
            # 扩展IR为单通道
            np_ir = np.expand_dims(np_ir, axis=-1)
            
            # 合并RGBT
            np_combined = np.concatenate([np_rgb, np_ir], axis=-1)
            pil_img = Image.fromarray(np_combined)
        else:
            print(f"Warning: IR image not found: {ir_path}, using RGB only")
            pil_img = img_rgb
    
    elif args.modality == 'ir':
        # 纯IR模式
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return None, None
        # IR图像转换为RGB（3通道），与dataloader保持一致
        pil_img = Image.open(img_path).convert('RGB')
    
    else:  # RGB模式
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
    args.visual_output_dir = args.output_dir
    
    # 构建变换
    transform = make_transforms(args, 'val')
    
    # 处理每个样本
    success_count = 0
    fail_count = 0
    
    for idx, item in enumerate(samples_to_process):
        sample_idx = args.start_idx + idx
        
        # 解析数据格式: [image_filename, img_id, bbox(xywh), text]
        img_filename = item[0]
        bbox_gt = item[2]  # ground truth bbox
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
                pred_box, pred_seg, img_cls, text_cls = model(img_nt.tensors, img_nt.mask, texts)
            
            # 可视化结果
            img_t = img_tensor[0].cpu()
            bbox = pred_box[0].cpu()
            
            vis_img, out_path = visualize_oneref_pred_only(
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
    parser = argparse.ArgumentParser('OneRef Visualization', parents=[get_args_parser()])
    args = parser.parse_args()
    
    visualize_dataset(args)


if __name__ == '__main__':
    main()
