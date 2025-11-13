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


def visualize_gt_only(args):
    """只可视化GT标注"""
    print("Starting GT-only visualization...")
    
    # 加载数据集
    dataset = load_dataset(args.label_file)
    
    # 选择要可视化的样本
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    samples_to_process = dataset[args.start_idx:end_idx]
    print(f"Visualizing GT for samples {args.start_idx} to {end_idx - 1} (total: {len(samples_to_process)})")
    
    # 构建变换（用于图像预处理）
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
            if result is None:
                fail_count += 1
                continue
            
            # 根据模态解析返回值
            if args.modality == 'rgbt':
                if len(result) != 4:
                    fail_count += 1
                    continue
                img_tensor, img_mask, pil_img_original, pil_img_ir = result
            else:
                if len(result) != 3:
                    fail_count += 1
                    continue
                img_tensor, img_mask, pil_img_original = result
                pil_img_ir = None
            
            if img_tensor is None:
                fail_count += 1
                continue
            
            # 保存GT可视化
            gt_path = save_gt_visualization(
                args, pil_img_original, pil_img_ir, text, bbox_gt,
                sample_idx=sample_idx,
                output_dir=args.output_dir,
                model_name="gt"
            )
            
            success_count += 1
            if (idx + 1) % 10 == 0:
                print(f"Progress: {idx + 1}/{len(samples_to_process)} - Success: {success_count}, Failed: {fail_count}")
        
        except Exception as e:
            print(f"Error processing sample {sample_idx} ({img_filename}): {str(e)}")
            fail_count += 1
            continue
    
    print(f"\nGT visualization complete!")
    print(f"Total processed: {len(samples_to_process)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"GT results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser('GT Visualization', parents=[get_args_parser()])
    args = parser.parse_args()
    
    visualize_gt_only(args)


if __name__ == '__main__':
    main()
