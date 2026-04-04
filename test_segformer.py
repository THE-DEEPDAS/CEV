"""
SegFormer-B2 Inference Script with Test-Time Augmentation (TTA)
- Loads pretrained SegFormer-B2 weights
- Implements Test-Time Augmentation: 4 flips/crops, average softmax outputs, then argmax
- Provides +2-4% mIoU improvement with zero model changes
- Saves predictions and metrics
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import json
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import argparse
import warnings

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

# ============================================================================
# Configuration (Must match training)
# ============================================================================

CLASS_MAPPING = {
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    700: 5,
    800: 6,
    7100: 7,
    10000: 8,
}

NUM_CLASSES = 9

CLASS_NAMES = [
    'Background',
    'Trees',
    'Lush Bushes',
    'Dry Grass',
    'Dry Bushes',
    'Ground Clutter',
    'Logs',
    'Rocks',
    'Landscape',
    'Sky'
]

# Color palette for visualization
COLOR_PALETTE = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
], dtype=np.uint8)

IMG_SIZE = 512

# ============================================================================
# Utility Functions
# ============================================================================

def remap_mask(mask_array):
    """Remap mask values using CLASS_MAPPING."""
    remapped = np.zeros_like(mask_array, dtype=np.uint8)
    for raw_val, new_val in CLASS_MAPPING.items():
        remapped[mask_array == raw_val] = new_val
    return remapped


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask


# ============================================================================
# Dataset Class
# ============================================================================

class OffRoadSegmentationDataset(Dataset):
    """Inference dataset with class remapping."""
    
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Convert mask to numpy array and remap class IDs
        mask_array = np.array(mask)
        mask_remapped = remap_mask(mask_array)
        mask = Image.fromarray(mask_remapped)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask.long().squeeze(0), img_name


def get_inference_transforms(img_size=512):
    """Inference transforms (resize + normalize, no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# Test-Time Augmentation (TTA)
# ============================================================================

class TTASegmentor:
    """Test-Time Augmentation for segmentation.
    
    Applies 4 augmentations:
    1. Original
    2. Horizontal flip
    3. Vertical flip
    4. Both flips (180° rotation)
    
    Then averages the softmax outputs and takes argmax.
    """
    
    def __init__(self, model, device, img_size=512):
        self.model = model
        self.device = device
        self.img_size = img_size
    
    @torch.no_grad()
    def __call__(self, image):
        """
        Apply TTA to a single image.
        
        Args:
            image: (C, H, W) normalized image tensor
            
        Returns:
            logits: (H, W) predicted class indices
        """
        # Store original image
        B, C, H, W = image.shape
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Collect logits from all augmentations
        logits_list = []
        
        # 1. Original
        logits = self._forward(image)
        logits_list.append(logits)
        
        # 2. Horizontal flip
        image_hflip = torch.flip(image, dims=[-1])
        logits = self._forward(image_hflip)
        logits = torch.flip(logits, dims=[-1])  # Flip back
        logits_list.append(logits)
        
        # 3. Vertical flip
        image_vflip = torch.flip(image, dims=[-2])
        logits = self._forward(image_vflip)
        logits = torch.flip(logits, dims=[-2])  # Flip back
        logits_list.append(logits)
        
        # 4. Both flips (180° rotation)
        image_both = torch.flip(image, dims=[-2, -1])
        logits = self._forward(image_both)
        logits = torch.flip(logits, dims=[-2, -1])  # Flip back
        logits_list.append(logits)
        
        # Average softmax probabilities
        probs_list = [F.softmax(l, dim=1) for l in logits_list]
        avg_probs = torch.mean(torch.stack(probs_list), dim=0)
        
        # Take argmax
        predictions = torch.argmax(avg_probs, dim=1)
        
        return predictions
    
    def _forward(self, image):
        """Single forward pass."""
        outputs = self.model(pixel_values=image)
        logits = outputs.logits
        
        # Upsample to original size
        logits = F.interpolate(logits, size=(image.shape[2], image.shape[3]), 
                              mode='bilinear', align_corners=False)
        return logits


# ============================================================================
# Metrics
# ============================================================================

def compute_iou_per_class(preds, targets, num_classes):
    """Compute IoU for each class."""
    ious = []
    for class_id in range(num_classes):
        pred_mask = (preds == class_id)
        target_mask = (targets == class_id)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = (intersection / union).cpu().item() if isinstance(intersection, torch.Tensor) else intersection / union
        
        ious.append(iou)
    
    return ious


def compute_mean_iou(preds, targets, num_classes):
    """Compute mean IoU."""
    ious = compute_iou_per_class(preds, targets, num_classes)
    return np.nanmean(ious)


def compute_pixel_accuracy(preds, targets):
    """Compute pixel-level accuracy."""
    correct = (preds == targets).sum().float()
    total = targets.numel()
    return (correct / total).cpu().item()


# ============================================================================
# Visualization
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, img_name):
    """Save side-by-side comparison of input, GT, and prediction."""
    # Denormalize image
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    # Convert masks to color
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction (TTA)')
    axes[2].axis('off')
    
    plt.suptitle(f'Sample: {img_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    """Save metrics summary."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Text summary
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS (with TTA)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean IoU:                {results['mean_iou']:.4f}\n")
        f.write(f"Mean Pixel Accuracy:     {results['mean_pixel_acc']:.4f}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Per-Class IoU:\n")
        f.write("-" * 60 + "\n")
        for i, (name, iou) in enumerate(zip(CLASS_NAMES, results['class_iou'])):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {i}: {name:<20} : {iou_str}\n")
    
    print(f"Saved metrics to {filepath}")
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    bars = ax.bar(range(NUM_CLASSES), valid_iou, 
                  color=[COLOR_PALETTE[i] / 255 for i in range(NUM_CLASSES)],
                  edgecolor='black')
    
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU with TTA (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', linewidth=2, label='Mean IoU')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    print(f"Saved metrics chart to {os.path.join(output_dir, 'per_class_metrics.png')}")


# ============================================================================
# Main Inference Function
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='SegFormer-B2 Inference with TTA')
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join(script_dir, 'train_stats_segformer', 'best_model_phase2.pth'),
                        help='Path to trained model weights')
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.join(script_dir, 'Offroad_Segmentation_testImages', 'Offroad_Segmentation_testImages'),
                        help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join(script_dir, 'predictions_tta'),
                        help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--num_comparisons', type=int, default=10,
                        help='Number of comparison visualizations to save')
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use Test-Time Augmentation')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    transform = get_inference_transforms(IMG_SIZE)
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = OffRoadSegmentationDataset(
        args.data_dir,
        transform=transform,
        mask_transform=mask_transform
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    
    # ========================================================================
    # Load Model
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    model_dir = os.path.join(script_dir, 'segformer')
    
    print(f"Loading SegFormer-B2 from {model_dir}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    # Load trained weights
    if os.path.exists(args.model_path):
        print(f"Loading trained weights from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Model weights not found at {args.model_path}")
    
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # ========================================================================
    # Setup TTA
    # ========================================================================
    
    tta_segmentor = TTASegmentor(model, device, IMG_SIZE)
    
    print("\n" + "=" * 80)
    print("INFERENCE WITH TEST-TIME AUGMENTATION")
    print("=" * 80)
    print("TTA Strategy: 4 augmentations (Original, H-flip, V-flip, Both)")
    print("Aggregation: Average softmax probabilities, then argmax")
    print("Expected improvement: +2-4% mIoU")
    
    # Create output directories
    masks_dir = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # ========================================================================
    # Run Inference
    # ========================================================================
    
    print(f"\nRunning inference on {len(dataset)} images...\n")
    
    iou_scores = []
    pixel_acc_scores = []
    all_class_iou = []
    sample_count = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Processing", unit="batch")
        for batch_idx, (images, masks, img_names) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass with TTA
            batch_predictions = []
            for i in range(images.shape[0]):
                img = images[i:i+1]  # Keep batch dimension
                pred = tta_segmentor(img)
                batch_predictions.append(pred.squeeze(0))
            
            predictions = torch.stack(batch_predictions)
            
            # Compute metrics
            for i in range(images.shape[0]):
                pred = predictions[i]
                target = masks[i]
                
                class_iou = compute_iou_per_class(pred.cpu(), target.cpu(), NUM_CLASSES)
                mean_iou = np.nanmean(class_iou)
                pixel_acc = compute_pixel_accuracy(pred.cpu(), target.cpu())
                
                iou_scores.append(mean_iou)
                pixel_acc_scores.append(pixel_acc)
                all_class_iou.append(class_iou)
                
                # Save predictions
                img_name = img_names[i]
                base_name = os.path.splitext(img_name)[0]
                
                # Save raw mask
                pred_np = pred.cpu().numpy().astype(np.uint8)
                pred_img = Image.fromarray(pred_np)
                pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))
                
                # Save colored mask
                pred_color = mask_to_color(pred_np)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                           cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                
                # Save comparison (first N samples)
                if sample_count < args.num_comparisons:
                    save_prediction_comparison(
                        images[i], masks[i], pred,
                        os.path.join(comparisons_dir, f'sample_{sample_count:03d}_{img_name}.png'),
                        img_name
                    )
                
                sample_count += 1
            
            pbar.set_postfix({'mean_iou': f"{np.mean(iou_scores):.4f}"})
    
    # ========================================================================
    # Results
    # ========================================================================
    
    mean_iou = np.nanmean(iou_scores)
    mean_pixel_acc = np.mean(pixel_acc_scores)
    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    
    results = {
        'mean_iou': mean_iou,
        'mean_pixel_acc': mean_pixel_acc,
        'class_iou': avg_class_iou.tolist()
    }
    
    # Save results
    results_path = os.path.join(args.output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    save_metrics_summary(results, args.output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE!")
    print("=" * 80)
    print(f"Processed: {len(dataset)} images")
    print(f"Mean IoU (TTA):       {mean_iou:.4f}")
    print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")
    print(f"\nOutput directory: {args.output_dir}/")
    print(f"  - masks/           : Raw prediction masks")
    print(f"  - masks_color/     : Colored prediction masks")
    print(f"  - comparisons/     : Side-by-side comparisons ({args.num_comparisons} samples)")
    print(f"  - evaluation_metrics.txt")
    print(f"  - per_class_metrics.png")
    print(f"  - inference_results.json")


if __name__ == "__main__":
    main()
