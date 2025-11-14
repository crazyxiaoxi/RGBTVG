#!/usr/bin/env python3
"""
GT可视化脚本
专门用于保存数据集的Ground Truth可视化结果
"""
import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import make_transforms
from utils_visualization import process_image, save_gt_visualization, load_dataset

def get_args_parser():
    parser = argparse.ArgumentParser('GT Visualization', add_help=False)
    
    # 基本参数
    parser.add_argument('--label_file', required=True, type=str, help='数据标注文件路径')
    parser.add_argument('--dataroot', required=True, type=str, help='图像数据根目录')
    parser.add_argument('--output_dir', default='./visual_result/gt_only', type=str, help='输出目录')
    
    # 数据集参数
    parser.add_argument('--dataset', default='rgbtvg_flir', type=str, help='数据集名称')
    parser.add_argument('--modality', default='rgb', type=str, choices=['rgb', 'ir', 'rgbt'], help='图像模态')
    parser.add_argument('--num_samples', default=100, type=int, help='可视化样本数量')
    parser.add_argument('--start_idx', default=0, type=int, help='起始索引')
    
    # 图像参数
    parser.add_argument('--imsize', default=224, type=int, help='图像尺寸')
    
    return parser


def generate_annotation_statistics(output_dir, annotation_stats, dataset, modality):
    """生成annotation统计报告"""
    from pathlib import Path
    
    # 排序：按annotation数量降序排列
    annotation_stats.sort(key=lambda x: x['annotations'], reverse=True)
    
    # 计算统计信息
    total_images = len(annotation_stats)
    total_annotations = sum(item['annotations'] for item in annotation_stats)
    avg_annotations = total_annotations / total_images if total_images > 0 else 0
    
    # 统计annotation数量分布
    annotation_counts = {}
    for item in annotation_stats:
        count = item['annotations']
        annotation_counts[count] = annotation_counts.get(count, 0) + 1
    
    # 保存统计报告
    stats_path = os.path.join(output_dir, "annotation_statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GT ANNOTATION STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Modality: {modality}\n")
        f.write(f"Generated: {Path().absolute()}\n\n")
        
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Total Annotations: {total_annotations}\n")
        f.write(f"Average Annotations per Image: {avg_annotations:.2f}\n\n")
        
        f.write("ANNOTATION COUNT DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        for count in sorted(annotation_counts.keys()):
            images_with_count = annotation_counts[count]
            percentage = (images_with_count / total_images) * 100
            f.write(f"{count} annotations: {images_with_count} images ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("DETAILED LIST (sorted by annotation count):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<6} {'Image':<50} {'Annotations':<12}\n")
        f.write("-" * 80 + "\n")
        
        for i, item in enumerate(annotation_stats, 1):
            f.write(f"{i:<6} {item['image']:<50} {item['annotations']:<12}\n")
    
    # 同时生成CSV格式的统计文件
    csv_path = os.path.join(output_dir, "annotation_statistics.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Rank,Image,Annotations\n")
        for i, item in enumerate(annotation_stats, 1):
            f.write(f"{i},{item['image']},{item['annotations']}\n")
    
    print(f"\n📊 Annotation Statistics:")
    print(f"   Total Images: {total_images}")
    print(f"   Total Annotations: {total_annotations}")
    print(f"   Average per Image: {avg_annotations:.2f}")
    print(f"   Max Annotations: {max(item['annotations'] for item in annotation_stats) if annotation_stats else 0}")
    print(f"   Min Annotations: {min(item['annotations'] for item in annotation_stats) if annotation_stats else 0}")


def visualize_gt_only(args):
    """只可视化GT标注，按图片分组处理"""
    print("Starting GT-only visualization...")
    
    # 加载数据集
    dataset = load_dataset(args.label_file)
    
    # 选择要可视化的样本
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    samples_to_process = dataset[args.start_idx:end_idx]
    print(f"Visualizing GT for samples {args.start_idx} to {end_idx - 1} (total: {len(samples_to_process)})")
    
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
    
    print(f"Found {len(image_groups)} unique images with GT annotations")
    
    # 构建变换
    transform = make_transforms(args, 'val')
    
    # 处理每个图片组
    success_count = 0
    fail_count = 0
    processed_images = 0
    annotation_stats = []  # 用于统计每个图片的annotation数量
    
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
            
            # 保存合并的GT可视化
            save_combined_gt_visualization(
                args, pil_img_original, pil_img_ir, group_items, 
                img_filename, args.output_dir
            )
            
            # 记录统计信息
            annotation_stats.append({
                'image': img_filename,
                'annotations': len(group_items)
            })
            
            success_count += len(group_items)
            print(f"Processed image {processed_images}/{len(image_groups)}: {img_filename} ({len(group_items)} annotations)")
        
        except Exception as e:
            print(f"Error processing image {img_filename}: {str(e)}")
            fail_count += len(group_items)
            continue
    
    # 生成统计报告
    generate_annotation_statistics(args.output_dir, annotation_stats, args.dataset, args.modality)
    
    print(f"\nGT visualization complete!")
    print(f"Total images processed: {processed_images}")
    print(f"Total annotations: {len(samples_to_process)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"GT results saved to: {args.output_dir}")
    print(f"Statistics report saved to: {args.output_dir}/annotation_statistics.txt")


def save_combined_gt_visualization(args, pil_img_original, pil_img_ir, group_items, img_filename, output_dir):
    """保存合并的GT可视化结果（每个图片以文件夹形式存储）"""
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
    
    # 生成不同颜色用于区分不同的GT框
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
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
    
    # 3. 保存RGB图 + 所有GT框
    vis_img_rgb = np.ascontiguousarray(img_np.copy())
    
    # 收集所有文本
    all_texts = []
    
    for i, item in enumerate(group_items):
        bbox_gt = item['bbox_gt']
        text = item['text']
        sample_idx = item['sample_idx']
        
        # 处理bbox
        if isinstance(bbox_gt, torch.Tensor):
            bbox_gt = bbox_gt.cpu().numpy()
        elif isinstance(bbox_gt, list):
            bbox_gt = np.array(bbox_gt)
        
        if len(bbox_gt) == 4:
            gt_x, gt_y, gt_w, gt_h = bbox_gt.astype(int)
            gt_x_min = gt_x
            gt_y_min = gt_y
            gt_x_max = gt_x + gt_w
            gt_y_max = gt_y + gt_h
        else:
            print(f"Warning: Unexpected gt_bbox format: {bbox_gt}")
            continue
        
        # 限制在图像范围内
        gt_x_min = max(0, min(gt_x_min, w - 1))
        gt_y_min = max(0, min(gt_y_min, h - 1))
        gt_x_max = max(0, min(gt_x_max, w - 1))
        gt_y_max = max(0, min(gt_y_max, h - 1))
        
        # 选择颜色
        color = colors[i % len(colors)]
        
        # 画框
        cv2.rectangle(vis_img_rgb, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), color, 2)
        
        # 添加标签（框的编号）
        cv2.putText(vis_img_rgb, f"{i+1}", (gt_x_min, gt_y_min-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 收集文本
        all_texts.append(f"{i+1}. {text}")
    
    # 保存RGB图 + GT框
    rgb_gt_path = os.path.join(img_folder, "rgb_with_gt.jpg")
    cv2.imwrite(rgb_gt_path, cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR))
    
    # 4. 对于RGBT模态，保存IR图 + 所有GT框
    if hasattr(args, 'modality') and args.modality == 'rgbt' and pil_img_ir is not None:
        img_ir_np = np.array(pil_img_ir)
        
        if img_ir_np.ndim == 2:
            img_ir_np = np.stack([img_ir_np] * 3, axis=-1)
        elif img_ir_np.ndim == 3 and img_ir_np.shape[2] == 1:
            img_ir_np = np.repeat(img_ir_np, 3, axis=2)
        
        vis_img_ir = np.ascontiguousarray(img_ir_np.copy())
        
        # 画所有GT框到IR图上
        for i, item in enumerate(group_items):
            bbox_gt = item['bbox_gt']
            
            if isinstance(bbox_gt, torch.Tensor):
                bbox_gt = bbox_gt.cpu().numpy()
            elif isinstance(bbox_gt, list):
                bbox_gt = np.array(bbox_gt)
            
            if len(bbox_gt) == 4:
                gt_x, gt_y, gt_w, gt_h = bbox_gt.astype(int)
                gt_x_min = max(0, min(gt_x, w - 1))
                gt_y_min = max(0, min(gt_y, h - 1))
                gt_x_max = max(0, min(gt_x + gt_w, w - 1))
                gt_y_max = max(0, min(gt_y + gt_h, h - 1))
                
                color = colors[i % len(colors)]
                cv2.rectangle(vis_img_ir, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), color, 2)
                cv2.putText(vis_img_ir, f"{i+1}", (gt_x_min, gt_y_min-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        ir_gt_path = os.path.join(img_folder, "ir_with_gt.jpg")
        cv2.imwrite(ir_gt_path, cv2.cvtColor(vis_img_ir, cv2.COLOR_RGB2BGR))
    
    # 5. 保存合并的文本文件
    txt_path = os.path.join(img_folder, "annotations.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Image: {img_filename}\n")
        f.write(f"Total annotations: {len(group_items)}\n")
        f.write("=" * 50 + "\n\n")
        for text_line in all_texts:
            f.write(text_line + "\n\n")
    
    return rgb_gt_path


def main():
    parser = argparse.ArgumentParser('GT Visualization', parents=[get_args_parser()])
    args = parser.parse_args()
    
    visualize_gt_only(args)


if __name__ == '__main__':
    main()
