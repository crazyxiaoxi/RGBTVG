#!/usr/bin/env python3
"""
HiVG Visualization Script
Batch visualization of prediction results based on dataset files (.pth)
"""
import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# HiVG model imports
from models import build_model
from datasets import make_transforms
from utils.misc import NestedTensor
from utils.visual_utils import visualization
import cv2
from utils.box_utils import xywh2xyxy

# Import common visualization utilities
from utils_visualization import process_image, save_pred_visualization, load_dataset, generate_prediction_statistics


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
    """Load HiVG model"""
    print(f"Loading model from: {args.model_checkpoint}")
    
    checkpoint = torch.load(args.model_checkpoint, map_location='cpu')

    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']

        ckpt_args.gpu_id = getattr(args, 'gpu_id', getattr(ckpt_args, 'gpu_id', '0'))
        ckpt_args.output_dir = getattr(args, 'output_dir', getattr(ckpt_args, 'output_dir', './visual_result/hivg'))
        ckpt_args.num_samples = getattr(args, 'num_samples', getattr(ckpt_args, 'num_samples', 0))
        ckpt_args.start_idx = getattr(args, 'start_idx', getattr(ckpt_args, 'start_idx', 0))
        ckpt_args.label_file = getattr(args, 'label_file', getattr(ckpt_args, 'label_file', ''))
        ckpt_args.dataroot = getattr(args, 'dataroot', getattr(ckpt_args, 'dataroot', ''))
        ckpt_args.model_checkpoint = args.model_checkpoint
        ckpt_args.device = getattr(args, 'device', getattr(ckpt_args, 'device', 'cuda'))

        args = ckpt_args
        print("Using model configuration from checkpoint (with visualization overrides)")
    
    # Adjust hidden_dim based on model type
    if args.model == "ViT-L/14" or args.model == "ViT-L/14@336px":
        args.vl_hidden_dim = 768
    
    # Build model
    model = build_model(args)
    
    # Load checkpoint
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys[:5]}...")  # Print only first 5
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}...")
    
    print(f"Model loaded from epoch: {checkpoint.get('epoch', 'N/A')}")
    
    model.to(args.device)
    model.eval()
    
    return model, args



def visualize_dataset(args):
    """Visualize dataset, process by image groups"""
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device(args.device)
    
    # Load model
    model, args = load_model(args)
    
    # Load dataset
    dataset = load_dataset(args.label_file)
    
    # Determine sample range to visualize
    end_idx = args.start_idx + args.num_samples if args.num_samples > 0 else len(dataset)
    end_idx = min(end_idx, len(dataset))
    samples_to_process = dataset[args.start_idx:end_idx]
    
    print(f"Visualizing samples {args.start_idx} to {end_idx-1} (total: {len(samples_to_process)})")
    
    # Group by image filename
    image_groups = {}
    for idx, item in enumerate(samples_to_process):
        sample_idx = args.start_idx + idx
        
        # Parse data format
        if str(args.dataset).startswith('rgbtvg'):
            img_filename = item[0]
            img_size = item[1]
            bbox_gt = item[2]
            text = item[3].lower()
            lighting = item[4] if len(item) > 4 else None
            scale_cls = item[5] if len(item) > 5 else None
        else:
            img_filename = item[0]
            bbox_gt = item[2]
            text = item[3].lower()
        
        # Group by image filename
        if img_filename not in image_groups:
            image_groups[img_filename] = []
        
        image_groups[img_filename].append({
            'sample_idx': sample_idx,
            'bbox_gt': bbox_gt,
            'text': text,
            'item': item
        })
    
    print(f"Found {len(image_groups)} unique images with annotations")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build transforms
    transform = make_transforms(args, 'val')
    
    # Process each image group
    success_count = 0
    fail_count = 0
    processed_images = 0
    prediction_stats = []  # For statistics of predictions per image
    
    for img_filename, group_items in image_groups.items():
        processed_images += 1
        img_path = os.path.join(args.dataroot, img_filename)
        
        try:
            # Use first sample to process image (all samples use same image)
            first_item = group_items[0]


            result = process_image(args, img_path, first_item['text'], transform)
            if result is None:
                fail_count += len(group_items)
                continue
            
            # Parse return value based on modality
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

            # Predict for each query
            predictions = []
            for item in group_items:
                text = item['text']
                
                # Prepare model input
                img_tensor_batch = img_tensor.unsqueeze(0).to(device)
                img_mask_batch = img_mask.unsqueeze(0).to(device)
                img_nt = NestedTensor(img_tensor_batch, img_mask_batch)
                texts = [text]
                
                # Model inference
                with torch.no_grad():
                    # HiVG model returns tuple: (pred_box, logits_per_text, logits_per_image, visu_token_similarity, seg_mask)
                    outputs = model(img_nt, texts)
                    pred_boxes = outputs[0]  # pred_box is first element
                bbox = pred_boxes[0].cpu()
                
                predictions.append({
                    'bbox': bbox,
                    'text': text,
                    'sample_idx': item['sample_idx']
                })
            
            # Save merged prediction visualization (single image, multiple boxes, numbered and color-coded)
            save_pred_visualization(
                args, pil_img_original, pil_img_ir, predictions,
                img_filename, args.output_dir, "hivg"
            )
            
            # Record statistics
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
    
    # Generate statistics report
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
