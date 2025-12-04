#!/usr/bin/env python3
"""
CLIP_VG visualization script for generating prediction outputs on datasets
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import build_model
from datasets import make_transforms
from utils.misc import NestedTensor
from models.clip import clip

from utils_visualization import process_image, save_pred_visualization, load_dataset, generate_prediction_statistics

def get_args_parser():
    parser = argparse.ArgumentParser('CLIP_VG Visualization', add_help=False)
    
    # Basic parameters
    parser.add_argument('--model_checkpoint', required=True, type=str, help='Model checkpoint path')
    parser.add_argument('--label_file', required=True, type=str, help='Label file path')
    parser.add_argument('--dataroot', required=True, type=str, help='Image data root')
    parser.add_argument('--output_dir', default='./visual_result/clip_vg', type=str, help='Output directory')
    
    # Dataset parameters
    parser.add_argument('--dataset', default='rgbtvg_flir', type=str, help='Dataset name')
    parser.add_argument('--modality', default='rgb', type=str, choices=['rgb', 'ir', 'rgbt'], help='Image modality')
    parser.add_argument('--num_samples', default=0, type=int, help='Number of samples to visualize (0 means full dataset)')
    parser.add_argument('--start_idx', default=0, type=int, help='Starting index')
    
    # Training parameters (needed to init model, unused during viz)
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
    
    # Augmentation options (not used during inference)
    parser.add_argument('--aug_blur', action='store_true')
    parser.add_argument('--aug_crop', action='store_true')
    parser.add_argument('--aug_scale', action='store_true')
    parser.add_argument('--aug_translate', action='store_true')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='CLIP_VG', help='Model name')
    parser.add_argument('--model', type=str, default='ViT-B/16', help='CLIP model type')
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    
    # Image parameters
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--emb_size', default=512, type=int, help='Embedding dimension')
    
    # Vision-Language Transformer
    parser.add_argument('--vl_dropout', default=0.1, type=float)
    parser.add_argument('--vl_nheads', default=8, type=int)
    parser.add_argument('--vl_hidden_dim', default=512, type=int)
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int)
    parser.add_argument('--vl_enc_layers', default=6, type=int)
    
    # Other parameters
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
    """Load CLIP_VG model"""
    print(f"Loading CLIP_VG model from: {args.model_checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    # Use checkpoint args if present
    if 'args' in checkpoint:
        print("Using model configuration from checkpoint...")
        model_args = checkpoint['args']
        # Preserve visualization arguments
        model_args.gpu_id = args.gpu_id
        model_args.output_dir = args.output_dir
        model_args.num_samples = args.num_samples
        model_args.start_idx = args.start_idx
        model_args.label_file = args.label_file
        model_args.dataroot = args.dataroot
        # Ensure eval_model exists
        if not hasattr(model_args, 'eval_model'):
            model_args.eval_model = getattr(args, 'eval_model', '')
        args = model_args  # Use checkpoint config

    # Ensure required args exist
    if not hasattr(args, 'eval_model'):
        args.eval_model = ''
    
    # Build model (returns model object only)
    model = build_model(args)
    model.to(device)
    
    # Load weights
    if 'model' in checkpoint:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print("Model loaded successfully!")
    return model, args  # Return updated args


# Helper utilities (load_dataset/process_image/save_visualization) live in utils_visualization.py


def visualize_dataset(args):
    """Visualize dataset grouped by image filename"""
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, args = load_model(args, device)
    
    # Load dataset
    dataset = load_dataset(args.label_file)
    
    # Determine sample range to visualize
    end_idx = args.start_idx + args.num_samples if args.num_samples > 0 else len(dataset)
    end_idx = min(end_idx, len(dataset))
    samples_to_process = dataset[args.start_idx:end_idx]
    
    print(f"Visualizing samples {args.start_idx} to {end_idx-1} (total: {len(samples_to_process)})")
    
    # Group samples by image filename
    image_groups = {}
    for idx, item in enumerate(samples_to_process):
        sample_idx = args.start_idx + idx
        
        # Parse dataset entry
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
        
        # Append to filename group
        if img_filename not in image_groups:
            image_groups[img_filename] = []
        
        image_groups[img_filename].append({
            'sample_idx': sample_idx,
            'bbox_gt': bbox_gt,
            'text': text,
            'item': item
        })
    
    print(f"Found {len(image_groups)} unique images with annotations")
    
    # Build transforms
    transform = make_transforms(args, 'val')
    
    # Process each image group
    success_count = 0
    fail_count = 0
    processed_images = 0
    prediction_stats = []  # Track number of predictions per image
    
    for img_filename, group_items in image_groups.items():
        processed_images += 1
        img_path = os.path.join(args.dataroot, img_filename)
        
        try:
            # Use first sample to process the shared image
            first_item = group_items[0]
            result = process_image(args, img_path, first_item['text'], transform)
            if result is None:
                fail_count += len(group_items)
                continue
            
            # Parse return values based on modality
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
                
                # Process text
                tokenizer = clip.tokenize([text], truncate=True).to(device)
                word_id_tensor = tokenizer
                word_mask_tensor = (word_id_tensor != 0).float()
                text_nt = NestedTensor(word_id_tensor, word_mask_tensor)
                
                # Model inference
                with torch.no_grad():
                    pred_boxes = model(img_nt, text_nt)
                bbox = pred_boxes[0].cpu()
                
                predictions.append({
                    'bbox': bbox,
                    'text': text,
                    'sample_idx': item['sample_idx']
                })
            
            # Save merged visualization (single image with multiple boxes)
            save_pred_visualization(
                args, pil_img_original, pil_img_ir, predictions,
                img_filename, args.output_dir, "clip_vg"
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
