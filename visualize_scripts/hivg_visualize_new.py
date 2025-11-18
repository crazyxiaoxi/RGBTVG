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

import numpy as np
from PIL import Image
from pathlib import Path


def load_dataset(label_file):
    """加载HiVG使用的数据集文件"""
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Dataset file not found: {label_file}")

    print(f"Loading dataset from: {label_file}")
    data = torch.load(label_file, map_location="cpu")
    print(f"Total samples in dataset: {len(data)}")
    return data


def process_image_hivg(args, img_path, text, transform):
    """HiVG 专用图像处理逻辑，模仿 dataloader 行为。

    返回:
        RGBT: (img_tensor, img_mask, pil_img_original, pil_img_ir)
        其他: (img_tensor, img_mask, pil_img_original)
    """
    pil_img_original = None
    pil_img_ir = None

    if args.modality == "rgbt":
        # 自动配对 RGB / IR
        if "/rgb/" in img_path:
            rgb_path = img_path
            ir_path = img_path.replace("/rgb/", "/ir/")
        elif "/ir/" in img_path:
            ir_path = img_path
            rgb_path = img_path.replace("/ir/", "/rgb/")
        else:
            rgb_path = img_path
            ir_path = img_path.replace("rgb", "ir")

        if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
            print(f"Warning: RGB or IR image not found: {rgb_path}, {ir_path}")
            return None, None, None, None

        img_rgb = Image.open(rgb_path).convert("RGB")
        img_ir = Image.open(ir_path)

        pil_img_original = img_rgb.copy()
        pil_img_ir = img_ir.copy()

        np_rgb = np.array(img_rgb)
        np_ir = np.array(img_ir)
        if np_ir.ndim == 3 and np_ir.shape[-1] == 3:
            np_ir = np_ir[..., 0]
        if np_ir.ndim == 2:
            np_ir = np.expand_dims(np_ir, axis=-1)
        np_combined = np.concatenate([np_rgb, np_ir], axis=-1)
        img = Image.fromarray(np_combined)

        w, h = img.size
        full_box_xyxy = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], dtype=torch.float32)

        input_dict = {"img": img, "box": full_box_xyxy, "text": text}
        input_dict = transform(input_dict)

        return input_dict["img"], input_dict["mask"], pil_img_original, pil_img_ir

    # 非 RGBT，直接读取为 RGB
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        return None, None, None

    pil_img = Image.open(img_path).convert("RGB")
    pil_img_original = pil_img.copy()

    w, h = pil_img.size
    full_box_xyxy = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], dtype=torch.float32)

    input_dict = {"img": pil_img, "box": full_box_xyxy, "text": text}
    input_dict = transform(input_dict)

    return input_dict["img"], input_dict["mask"], pil_img_original


def save_hivg_visualization(pil_img_original, predictions, img_filename, output_dir):
    """为 HiVG 保存单张图像上的多框预测结果。

    - 每张原图输出一张 RGB 图
    - 文件名与原图一致
    - 不做数据集特定的 Y 轴偏移
    - 使用多种颜色和编号区分不同 bbox
    """
    img_np = np.array(pil_img_original)
    h, w = img_np.shape[:2]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img_base_name = Path(img_filename).name
    save_path = os.path.join(output_dir, img_base_name)

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    vis_img_rgb = np.ascontiguousarray(img_np.copy())

    for i, pred in enumerate(predictions):
        bbox = pred.get("bbox", None)
        if bbox is None:
            continue

        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        elif isinstance(bbox, list):
            bbox = np.array(bbox)

        if len(bbox) != 4:
            print(f"Warning: Unexpected pred_bbox format: {bbox}")
            continue

        # 归一化 (xc, yc, w, h) -> 像素坐标，HiVG 不做额外偏移
        x_center, y_center, bw, bh = bbox
        print("debug!!!!!!", x_center, y_center, bw, bh)
        x_min = int((x_center - bw / 2) * w)
        y_min = int((y_center - bh / 2) * w)
        x_max = int((x_center + bw / 2) * w)
        y_max = int((y_center + bh / 2) * w)

        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, w - 1))
        x_max = max(0, min(x_max, w - 1))
        y_max = max(0, min(y_max, w - 1))

        color = colors[i % len(colors)]
        cv2.rectangle(vis_img_rgb, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(
            vis_img_rgb,
            f"{i+1}",
            (x_min, max(0, y_min - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.imwrite(save_path, cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR))
    return save_path


def generate_prediction_statistics(output_dir, prediction_stats, dataset, modality, model_name):
    """生成预测统计报告（复制自通用实现但独立于 utils_visualization）。"""
    prediction_stats.sort(key=lambda x: x["predictions"], reverse=True)

    total_images = len(prediction_stats)
    total_predictions = sum(item["predictions"] for item in prediction_stats)
    avg_predictions = total_predictions / total_images if total_images > 0 else 0

    prediction_counts = {}
    for item in prediction_stats:
        count = item["predictions"]
        prediction_counts[count] = prediction_counts.get(count, 0) + 1

    stats_path = os.path.join(output_dir, "prediction_statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
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

    csv_path = os.path.join(output_dir, "prediction_statistics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Rank,Image,Predictions\n")
        for i, item in enumerate(prediction_stats, 1):
            f.write(f"{i},{item['image']},{item['predictions']}\n")

    print(f"\n📊 {model_name.upper()} Prediction Statistics:")
    print(f"   Total Images: {total_images}")
    print(f"   Total Predictions: {total_predictions}")
    print(f"   Average per Image: {avg_predictions:.2f}")
    print(f"   Max Predictions: {max(item['predictions'] for item in prediction_stats) if prediction_stats else 0}")
    print(f"   Min Predictions: {min(item['predictions'] for item in prediction_stats) if prediction_stats else 0}")


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
    parser.add_argument('--num_samples', type=int, default=0,
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



def visualize_dataset(args):
    """可视化数据集，按图片分组处理"""
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device(args.device)
    
    # 加载模型
    model = load_model(args)
    
    # 加载数据集（本文件内的专用实现）
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
            result = process_image_hivg(args, img_path, first_item['text'], transform)
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
                    # HiVG模型返回tuple: (pred_box, logits_per_text, logits_per_image, visu_token_similarity, seg_mask)
                    outputs = model(img_nt, texts)
                    pred_boxes = outputs[0]  # pred_box是第一个元素
                
                bbox = pred_boxes[0].cpu()
                
                predictions.append({
                    'bbox': bbox,
                    'text': text,
                    'sample_idx': item['sample_idx']
                })
            
            # 保存合并的预测可视化（单图，多框，编号+颜色区分）
            save_hivg_visualization(
                pil_img_original, predictions,
                img_filename, args.output_dir,
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
    generate_prediction_statistics(args.output_dir, prediction_stats, args.dataset, args.modality, "hivg")
    
    print(f"\nVisualization complete!")
    print(f"Total images processed: {processed_images}")
    print(f"Total predictions: {len(samples_to_process)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Statistics report saved to: {args.output_dir}/prediction_statistics.txt")


def main():
    parser = argparse.ArgumentParser('HiVG Visualization', parents=[get_args_parser()])
    args = parser.parse_args()
    
    visualize_dataset(args)


if __name__ == '__main__':
    main()
