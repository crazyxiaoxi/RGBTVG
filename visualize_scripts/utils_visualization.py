"""
Shared visualization utilities for preprocessing, GT saving, and reporting
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path


def process_image(args, img_path, text, transform):
    """Process a single image, returning transformed tensors for inference and originals for visualization
    
    Args:
        args: configuration namespace with modality info
        img_path: path to the image
        text: text description
        transform: callable applied to the image
        
    Returns:
        For RGBT modality: (img_tensor, img_mask, pil_img_original, pil_img_ir)
        Otherwise: (img_tensor, img_mask, pil_img_original)
    """
    pil_img_original = None  # keep original RGB image for visualization
    pil_img_ir = None  # keep original IR image for visualization
    
    if args.modality == 'rgbt':
        # Try to automatically pair RGB and IR images
        if '/rgb/' in img_path:
            rgb_path = img_path
            ir_path = img_path.replace('/rgb/', '/ir/')
        elif '/ir/' in img_path:
            ir_path = img_path
            rgb_path = img_path.replace('/ir/', '/rgb/')
        else:
            rgb_path = img_path
            ir_path = img_path.replace('rgb', 'ir')
        
        # Verify both files exist
        if not os.path.exists(rgb_path) or not os.path.exists(ir_path):
            print(f"Warning: RGB or IR image not found: {rgb_path}, {ir_path}")
            return None, None, None, None
        
        # Load RGB and IR images exactly like the dataloader
        img_rgb_path = rgb_path
        img_ir_path = ir_path
        img_rgb = Image.open(img_rgb_path).convert("RGB")
        img_ir = Image.open(img_ir_path)
        pil_img_original = img_rgb.copy()
        pil_img_ir = img_ir.copy()

        # Mirror dataloader preprocessing
        np_rgb = np.array(img_rgb)
        np_ir = np.array(img_ir)
        if np_ir.shape[-1] == 3:
            np_ir = np_ir[..., 0]
        np_ir = np.expand_dims(np_ir, axis=-1)
        np_combined = np.concatenate([np_rgb, np_ir], axis=-1)
        img = Image.fromarray(np_combined)

        # Obtain image size
        w, h = img.size
        full_box_xyxy = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], dtype=torch.float32)

        # Apply transform to RGBT image
        input_dict = {'img': img, 'box': full_box_xyxy, 'text': text}
        input_dict = transform(input_dict)

        return input_dict['img'], input_dict['mask'], pil_img_original, pil_img_ir
        
    else:
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return None, None, None
        # Convert IR image to RGB (3 channels) to match dataloader
        pil_img = Image.open(img_path).convert('RGB')
        
        # Store original image for visualization
        pil_img_original = pil_img.copy()
        
        # Get image dimensions
        w, h = pil_img.size
        full_box_xyxy = torch.tensor([0.0, 0.0, float(w - 1), float(h - 1)], dtype=torch.float32)
        
        # Apply transform
        input_dict = {'img': pil_img, 'box': full_box_xyxy, 'text': text}
        input_dict = transform(input_dict)

        return input_dict['img'], input_dict['mask'], pil_img_original


def save_gt_visualization(args, pil_img_original, pil_img_ir, text, gt_bbox, sample_idx, output_dir, model_name="model"):
    """Save GT visualization (GT boxes only).
    
    Args:
        args: namespace containing modality/dataset config
        pil_img_original: original RGB PIL image
        pil_img_ir: optional IR image
        text: caption text
        gt_bbox: ground-truth bounding box
        sample_idx: sample index
        output_dir: directory for outputs
        model_name: used for file naming
        
    Returns:
        str: path to RGB visualization
    """
    img_np = np.array(pil_img_original)
    
    h, w = img_np.shape[:2]

    if isinstance(gt_bbox, torch.Tensor):
        gt_bbox = gt_bbox.cpu().numpy()
    elif isinstance(gt_bbox, list):
        gt_bbox = np.array(gt_bbox)
    
    if len(gt_bbox) == 4:
        gt_x, gt_y, gt_w, gt_h = gt_bbox.astype(int)
        gt_x_min = gt_x
        gt_y_min = gt_y
        gt_x_max = gt_x + gt_w
        gt_y_max = gt_y + gt_h
    else:
        print(f"Warning: Unexpected gt_bbox format: {gt_bbox}")
        gt_x_min = gt_y_min = gt_x_max = gt_y_max = 0
    
    gt_x_min = max(0, min(gt_x_min, w - 1))
    gt_y_min = max(0, min(gt_y_min, h - 1))
    gt_x_max = max(0, min(gt_x_max, w - 1))
    gt_y_max = max(0, min(gt_y_max, h - 1))
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Save original image (mainly for RGBT workflows)
    original_path = os.path.join(output_dir, f"{model_name}_{sample_idx:06d}_original.jpg")
    cv2.imwrite(original_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    # Save RGB image with GT box
    vis_img_rgb = np.ascontiguousarray(img_np)
    cv2.rectangle(vis_img_rgb, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (0, 0, 255), 2)  # Red GT box
    rgb_path = os.path.join(output_dir, f"{model_name}_{sample_idx:06d}_rgb.jpg")
    cv2.imwrite(rgb_path, cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR))
    
    # For RGBT, save IR image with GT box
    if hasattr(args, 'modality') and args.modality == 'rgbt' and pil_img_ir is not None:
        img_ir_np = np.array(pil_img_ir)

        if img_ir_np.ndim == 2:
            img_ir_np = np.stack([img_ir_np] * 3, axis=-1)
        elif img_ir_np.ndim == 3 and img_ir_np.shape[2] == 1:
            img_ir_np = np.repeat(img_ir_np, 3, axis=2)
        
        vis_img_ir = np.ascontiguousarray(img_ir_np)
        cv2.rectangle(vis_img_ir, (gt_x_min, gt_y_min), (gt_x_max, gt_y_max), (0, 0, 255), 2)
        ir_path = os.path.join(output_dir, f"{model_name}_{sample_idx:06d}_ir.jpg")
        cv2.imwrite(ir_path, cv2.cvtColor(vis_img_ir, cv2.COLOR_RGB2BGR))
        

    
    # Save caption text file
    txt_path = os.path.join(output_dir, f"{model_name}_{sample_idx:06d}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return rgb_path


def save_visualization(args, pil_img_original, pil_img_ir, text, pred_bbox, gt_bbox, sample_idx, output_dir, model_name="model"):
    """Save visualization outputs for predicted and ground-truth boxes."""
    # Save prediction visualization
    pred_rgb_path = save_pred_visualization(args, pil_img_original, pil_img_ir, text, pred_bbox, sample_idx, output_dir, model_name)
    
    # Save GT visualization
    gt_rgb_path = save_gt_visualization(args, pil_img_original, pil_img_ir, text, gt_bbox, sample_idx, output_dir, model_name)
    
    return pred_rgb_path, gt_rgb_path


def save_pred_visualization(args, pil_img_original, pil_img_ir,
                            text_or_predictions,
                            pred_bbox_or_img_filename,
                            sample_idx_or_output_dir,
                            output_dir_or_model_name,
                            model_name="model"):
    """Save prediction visualizations.

    Supports two usages:

    1) Legacy: single prediction box
       text_or_predictions -> caption text
       pred_bbox_or_img_filename -> prediction bbox (x_center, y_center, w, h) normalized
       sample_idx_or_output_dir -> sample index
       output_dir_or_model_name -> output directory
       model_name -> used for filenames

    2) Modern: multiple predictions on one image (current default)
       text_or_predictions -> list of {'bbox': ..., 'text': ...}
       pred_bbox_or_img_filename -> source image filename
       sample_idx_or_output_dir -> output directory
       output_dir_or_model_name -> optional model name
    """

    img_np = np.array(pil_img_original)
    h, w = img_np.shape[:2]

    # ---------- Modern usage: merge multiple boxes into one image ----------
    if isinstance(text_or_predictions, list) and len(text_or_predictions) > 0 and isinstance(text_or_predictions[0], dict):
        predictions = text_or_predictions
        img_filename = pred_bbox_or_img_filename
        output_dir = sample_idx_or_output_dir

        # Override model name if provided
        if isinstance(output_dir_or_model_name, str):
            model_name = output_dir_or_model_name

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Use original filename for output
        img_base_name = Path(img_filename).name
        save_path = os.path.join(output_dir, img_base_name)

        # Color palette to distinguish predictions
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
            bbox = pred.get('bbox', None)
            if bbox is None:
                continue

            # Support tensor/list/ndarray
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.cpu().numpy()
            elif isinstance(bbox, list):
                bbox = np.array(bbox)

            if len(bbox) != 4:
                print(f"Warning: Unexpected pred_bbox format: {bbox}")
                continue

            # Reuse original coordinate conversion logic
            pred_x_center, pred_y_center, pred_bbox_w, pred_bbox_h = bbox

            pred_x_min = int((pred_x_center - pred_bbox_w / 2) * w)
            pred_y_min = int((pred_y_center - pred_bbox_h / 2) * w)
            pred_x_max = int((pred_x_center + pred_bbox_w / 2) * w)
            pred_y_max = int((pred_y_center + pred_bbox_h / 2) * w)

            if hasattr(args, 'dataset'):
                if args.dataset == 'rgbtvg_mfad':
                    pred_y_min -= 160
                    pred_y_max -= 160
                elif args.dataset == 'rgbtvg_m3fd':
                    pred_y_min -= 128
                    pred_y_max -= 128
                elif args.dataset == 'rgbtvg_flir':
                    pred_y_min -= 64
                    pred_y_max -= 64

            pred_x_min = max(0, min(pred_x_min, w - 1))
            pred_y_min = max(0, min(pred_y_min, h - 1))
            pred_x_max = max(0, min(pred_x_max, w - 1))
            pred_y_max = max(0, min(pred_y_max, h - 1))


            color = colors[i % len(colors)]
            cv2.rectangle(vis_img_rgb, (pred_x_min, pred_y_min), (pred_x_max, pred_y_max), color, 2)
            cv2.putText(
                vis_img_rgb,
                f"{i+1}",
                (pred_x_min, pred_y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imwrite(save_path, cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR))
        return save_path

    # ---------- Legacy usage: single prediction, original behavior ----------
    text = text_or_predictions
    pred_bbox = pred_bbox_or_img_filename
    sample_idx = sample_idx_or_output_dir
    output_dir = output_dir_or_model_name

    if isinstance(pred_bbox, torch.Tensor):
        pred_bbox = pred_bbox.cpu().numpy()

    pred_x_center, pred_y_center, pred_bbox_w, pred_bbox_h = pred_bbox

    pred_x_min = int((pred_x_center - pred_bbox_w / 2) * w)
    pred_y_min = int((pred_y_center - pred_bbox_h / 2) * w)
    pred_x_max = int((pred_x_center + pred_bbox_w / 2) * w)
    pred_y_max = int((pred_y_center + pred_bbox_h / 2) * w)

    if hasattr(args, 'dataset'):
        if args.dataset == 'rgbtvg_mfad':
            pred_y_min -= 160
            pred_y_max -= 160
        elif args.dataset == 'rgbtvg_m3fd':
            pred_y_min -= 128
            pred_y_max -= 128
        elif args.dataset == 'rgbtvg_flir':
            pred_y_min -= 64
            pred_y_max -= 64

    pred_x_min = max(0, min(pred_x_min, w - 1))
    pred_y_min = max(0, min(pred_y_min, h - 1))
    pred_x_max = max(0, min(pred_x_max, w - 1))
    pred_y_max = max(0, min(pred_y_max, h - 1))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    vis_img_rgb = np.ascontiguousarray(img_np)
    cv2.rectangle(vis_img_rgb, (pred_x_min, pred_y_min), (pred_x_max, pred_y_max), (0, 255, 0), 2)
    rgb_path = os.path.join(output_dir, f"{model_name}_pred_{sample_idx:06d}_rgb.jpg")
    cv2.imwrite(rgb_path, cv2.cvtColor(vis_img_rgb, cv2.COLOR_RGB2BGR))

    if hasattr(args, 'modality') and args.modality == 'rgbt' and pil_img_ir is not None:
        img_ir_np = np.array(pil_img_ir)
        if img_ir_np.ndim == 2:
            img_ir_np = np.stack([img_ir_np] * 3, axis=-1)
        elif img_ir_np.ndim == 3 and img_ir_np.shape[2] == 1:
            img_ir_np = np.repeat(img_ir_np, 3, axis=2)

        vis_img_ir = np.ascontiguousarray(img_ir_np)
        cv2.rectangle(vis_img_ir, (pred_x_min, pred_y_min), (pred_x_max, pred_y_max), (0, 255, 0), 2)
        ir_path = os.path.join(output_dir, f"{model_name}_pred_{sample_idx:06d}_ir.jpg")
        cv2.imwrite(ir_path, cv2.cvtColor(vis_img_ir, cv2.COLOR_RGB2BGR))

    txt_path = os.path.join(output_dir, f"{model_name}_pred_{sample_idx:06d}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    return rgb_path


def load_dataset(label_file):
    """Load dataset pickle/pt file and return samples list."""
    import torch
    
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Dataset file not found: {label_file}")
    
    print(f"Loading dataset from: {label_file}")
    data = torch.load(label_file, map_location='cpu')
    print(f"Total samples in dataset: {len(data)}")
    return data


def generate_prediction_statistics(output_dir, prediction_stats, dataset, modality, model_name):
    """Generate prediction statistics report."""
    from pathlib import Path
    
    # Sort by prediction count descending
    prediction_stats.sort(key=lambda x: x['predictions'], reverse=True)
    
    # Compute summary statistics
    total_images = len(prediction_stats)
    total_predictions = sum(item['predictions'] for item in prediction_stats)
    avg_predictions = total_predictions / total_images if total_images > 0 else 0
    
    # Build prediction count distribution
    prediction_counts = {}
    for item in prediction_stats:
        count = item['predictions']
        prediction_counts[count] = prediction_counts.get(count, 0) + 1
    
    # Save statistics report
    stats_path = os.path.join(output_dir, "prediction_statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
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
    
    # Optionally generate CSV statistics
    csv_path = os.path.join(output_dir, "prediction_statistics.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Rank,Image,Predictions\n")
        for i, item in enumerate(prediction_stats, 1):
            f.write(f"{i},{item['image']},{item['predictions']}\n")
    
    print(f"\n📊 {model_name.upper()} Prediction Statistics:")
    print(f"   Total Images: {total_images}")
    print(f"   Total Predictions: {total_predictions}")
    print(f"   Average per Image: {avg_predictions:.2f}")
    print(f"   Max Predictions: {max(item['predictions'] for item in prediction_stats) if prediction_stats else 0}")
    print(f"   Min Predictions: {min(item['predictions'] for item in prediction_stats) if prediction_stats else 0}")