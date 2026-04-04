"""
SegFormer-B2 Training Script with Advanced Domain Generalization Techniques
- SegFormer-B2 backbone with pretrained weights
- Class ID remapping (100:0, 200:1, ..., 10000:9)
- Advanced data augmentation (ColorJitter, RandomCrop, GaussianBlur, etc.)
- Weighted CE + Dice Loss for handling class imbalance
- AdamW optimizer with polynomial LR decay and layer-wise LR
- Two-phase training: Phase 1 (freeze backbone), Phase 2 (finetune all)
- CutMix augmentation for rare classes
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import v2
from PIL import Image
import cv2
import os
import json
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import warnings
from contextlib import nullcontext

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

# ============================================================================
# Configuration
# ============================================================================

# Class mapping as requested
CLASS_MAPPING = {
    100: 0,      # Trees
    200: 1,      # Lush Bushes
    300: 2,      # Dry Grass
    500: 3,      # Dry Bushes
    550: 4,      # Ground Clutter (rare)
    700: 5,      # Logs (rare)
    800: 6,      # Rocks
    7100: 7,     # Landscape
    10000: 8,    # Sky
}

# Note: Background (0) maps to 0
NUM_CLASSES = 9

# Class weights for rare classes (Flowers removed, adjusted for actual classes)
# Ground Clutter (4), Logs (5) are rare
CLASS_WEIGHTS = torch.tensor([
    1.0,    # 0: Background
    1.0,    # 1: Trees
    1.0,    # 2: Lush Bushes
    1.0,    # 3: Dry Grass
    1.0,    # 4: Dry Bushes
    8.0,    # 5: Ground Clutter (rare)
    8.0,    # 6: Logs (rare)
    1.0,    # 7: Rocks
    0.5,    # 8: Sky (very common)
], dtype=torch.float32)

# Hyperparameters
# NOTE: On Windows + display-attached GPUs, large kernels can hit TDR timeout.
# Use micro-batches with gradient accumulation to preserve effective batch size.
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2  # Effective batch size = BATCH_SIZE * GRAD_ACCUM_STEPS = 2
IMG_SIZE = 512
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 40
LR_PHASE1 = 6e-5
LR_PHASE2 = 6e-5
WEIGHT_DECAY = 0.01
CE_WEIGHT = 0.7
DICE_WEIGHT = 0.3

print(f"NUM_CLASSES: {NUM_CLASSES}")
print(f"CLASS_MAPPING: {CLASS_MAPPING}")
print(f"CLASS_WEIGHTS: {CLASS_WEIGHTS}")
print(f"BATCH_SIZE: {BATCH_SIZE}, GRAD_ACCUM_STEPS: {GRAD_ACCUM_STEPS}, EFFECTIVE_BATCH_SIZE: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

# ============================================================================
# Utility Functions
# ============================================================================

def remap_mask(mask_array):
    """Remap mask values using CLASS_MAPPING. Input is raw pixel values."""
    remapped = np.zeros_like(mask_array, dtype=np.uint8)
    for raw_val, new_val in CLASS_MAPPING.items():
        remapped[mask_array == raw_val] = new_val
    # Background (0 -> 0) stays as is
    return remapped


# ============================================================================
# Data Augmentation Functions
# ============================================================================

class CutMix:
    """CutMix augmentation for rare classes. Pastes patches of rare classes into training images."""
    
    def __init__(self, rare_class_ids=[4, 5], prob=0.5, patch_size=64):
        """
        Args:
            rare_class_ids: List of rare class IDs (Ground Clutter=4, Logs=5)
            prob: Probability of applying CutMix
            patch_size: Size of patches to cut and paste
        """
        self.rare_class_ids = rare_class_ids
        self.prob = prob
        self.patch_size = patch_size
    
    def __call__(self, image, mask):
        """Apply CutMix to both image and mask."""
        if np.random.rand() > self.prob:
            return image, mask
        
        h, w = mask.shape[:2] if len(mask.shape) == 3 else mask.shape
        
        # Find pixels of rare classes in the batch
        for rare_class in self.rare_class_ids:
            if (mask == rare_class).sum() == 0:
                continue
            
            # Get coordinates of rare class
            rare_coords = np.where(mask == rare_class)
            if len(rare_coords[0]) == 0:
                continue
            
            # Choose a source center pixel belonging to the rare class
            src_center_y = int(np.random.choice(rare_coords[0]))
            src_center_x = int(np.random.choice(rare_coords[1]))

            # Build a source patch around the center, clamped to image bounds
            src_y0 = max(0, src_center_y - self.patch_size // 2)
            src_x0 = max(0, src_center_x - self.patch_size // 2)
            src_y1 = min(h, src_y0 + self.patch_size)
            src_x1 = min(w, src_x0 + self.patch_size)
            src_y0 = max(0, src_y1 - self.patch_size)
            src_x0 = max(0, src_x1 - self.patch_size)

            patch_h = src_y1 - src_y0
            patch_w = src_x1 - src_x0
            if patch_h <= 0 or patch_w <= 0:
                continue

            # Random target location that can fit the exact source patch size
            tgt_y0 = np.random.randint(0, max(1, h - patch_h + 1))
            tgt_x0 = np.random.randint(0, max(1, w - patch_w + 1))
            tgt_y1 = tgt_y0 + patch_h
            tgt_x1 = tgt_x0 + patch_w

            # Copy source patch and paste to target (shapes always match)
            if len(image.shape) == 3:
                src_img_patch = image[src_y0:src_y1, src_x0:src_x1].copy()
                image[tgt_y0:tgt_y1, tgt_x0:tgt_x1] = src_img_patch

            src_mask_patch = mask[src_y0:src_y1, src_x0:src_x1].copy()
            mask[tgt_y0:tgt_y1, tgt_x0:tgt_x1] = src_mask_patch
        
        return image, mask


def get_train_transforms(img_size=512):
    """Training transforms with emphasis on color jitter and domain generalization."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(img_size=512):
    """Validation transforms (no augmentation, just resize + normalize)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# Dataset Class
# ============================================================================

class OffRoadSegmentationDataset(Dataset):
    """Dataset with class remapping and optional CutMix augmentation."""
    
    def __init__(self, data_dir, transform=None, mask_transform=None, apply_cutmix=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.apply_cutmix = apply_cutmix
        
        if self.apply_cutmix:
            self.cutmix = CutMix(rare_class_ids=[4, 5], prob=0.3, patch_size=64)
        
        # Get list of image files
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
        
        # Apply CutMix to both image and mask (before standard transforms)
        if self.apply_cutmix:
            img_array = np.array(image)
            mask_array = np.array(mask)
            img_array, mask_array = self.cutmix(img_array, mask_array)
            image = Image.fromarray(img_array)
            mask = Image.fromarray(mask_array)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask.long().squeeze(0)


# ============================================================================
# Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation."""
    
    def __init__(self, smooth=1.0, ignore_index=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) model output
            targets: (B, H, W) target class indices
        """
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        target_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        
        # Compute dice for each class
        intersection = (probs * target_one_hot).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        return dice_loss.mean()


class WeightedCombinedLoss(nn.Module):
    """Combination of Weighted CrossEntropyLoss and Dice Loss."""
    
    def __init__(self, ce_weight=0.7, dice_weight=0.3, class_weights=None, ignore_index=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if ignore_index is None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=int(ignore_index))
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
    
    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


# ============================================================================
# Metrics
# ============================================================================

def compute_iou_per_class(logits, targets, num_classes, ignore_index=None):
    """Compute IoU for each class."""
    preds = torch.argmax(logits, dim=1)
    
    ious = []
    for class_id in range(num_classes):
        pred_mask = (preds == class_id)
        target_mask = (targets == class_id)
        
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = (intersection / union).cpu().item()
        
        ious.append(iou)
    
    return ious


def compute_mean_iou(logits, targets, num_classes):
    """Compute mean IoU."""
    ious = compute_iou_per_class(logits, targets, num_classes)
    return np.mean(ious)


def compute_pixel_accuracy(logits, targets):
    """Compute pixel-level accuracy."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().float()
    total = targets.numel()
    return (correct / total).cpu().item()


# ============================================================================
# Polynomial LR Scheduler
# ============================================================================

def polynomial_decay_scheduler(optimizer, num_epochs, base_lr, power=1.0):
    """Polynomial decay learning rate scheduler."""
    def lr_lambda(epoch):
        return (1 - epoch / num_epochs) ** power
    
    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Optimizer with Layer-wise LR
# ============================================================================

def create_optimizer_with_layer_wise_lr(model, base_lr, weight_decay):
    """Create AdamW optimizer with layer-wise learning rates.
    Backbone at 1x, head at 10x.
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Hugging Face SegFormer backbone params live under `segformer.encoder`
        if name.startswith("segformer.encoder"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': base_lr},
        {'params': head_params, 'lr': base_lr * 10},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    
    return optimizer


# ============================================================================
# Training Function
# ============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device, phase_name="Train"):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    pbar = tqdm(train_loader, desc=f"{phase_name}", leave=False)
    optimizer.zero_grad()
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward pass
        autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp) if use_amp else nullcontext()
        with autocast_ctx:
            outputs = model(pixel_values=images)
            logits = outputs.logits
        
            # Upsample to original size
            logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
        
            # Compute loss
            loss = loss_fn(logits, masks)
            loss_for_step = loss / GRAD_ACCUM_STEPS
        
        # Backward pass
        scaler.scale(loss_for_step).backward()

        # Use non-foreach gradient clipping to avoid some CUDA timeout issues on Windows.
        if ((batch_idx + 1) % GRAD_ACCUM_STEPS == 0) or ((batch_idx + 1) == len(train_loader)):
            try:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            except RuntimeError as e:
                if "launch timed out" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += float(loss.detach().cpu().item())
        with torch.no_grad():
            total_iou += compute_mean_iou(logits, masks, NUM_CLASSES)
            total_acc += compute_pixel_accuracy(logits, masks)
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'iou': f"{total_iou/num_batches:.4f}"
        })
    
    return {
        'loss': total_loss / num_batches,
        'iou': total_iou / num_batches,
        'acc': total_acc / num_batches
    }


@torch.no_grad()
def validate(model, val_loader, loss_fn, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        outputs = model(pixel_values=images)
        logits = outputs.logits
        
        # Upsample to original size
        logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
        
        # Compute loss
        loss = loss_fn(logits, masks)
        
        # Metrics
        total_loss += loss.item()
        total_iou += compute_mean_iou(logits, masks, NUM_CLASSES)
        total_acc += compute_pixel_accuracy(logits, masks)
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return {
        'loss': total_loss / num_batches,
        'iou': total_iou / num_batches,
        'acc': total_acc / num_batches
    }


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats_segformer')
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # Setup Data
    # ========================================================================
    
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    train_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'val')
    
    # Transforms
    train_transform = get_train_transforms(IMG_SIZE)
    val_transform = get_val_transforms(IMG_SIZE)
    
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Datasets
    train_dataset = OffRoadSegmentationDataset(
        train_dir, 
        transform=train_transform, 
        mask_transform=mask_transform,
        apply_cutmix=True
    )
    val_dataset = OffRoadSegmentationDataset(
        val_dir,
        transform=val_transform,
        mask_transform=mask_transform,
        apply_cutmix=False
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    
    # ========================================================================
    # Setup Model
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    # Load SegFormer-B2 from local pretrained weights
    model_dir = os.path.join(script_dir, 'segformer')
    
    print(f"Loading SegFormer-B2 from {model_dir}...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========================================================================
    # Setup Loss and Optimizer
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TRAINING SETUP")
    print("=" * 80)
    
    CLASS_WEIGHTS_DEVICE = CLASS_WEIGHTS.to(device)
    loss_fn = WeightedCombinedLoss(
        ce_weight=CE_WEIGHT,
        dice_weight=DICE_WEIGHT,
        class_weights=CLASS_WEIGHTS_DEVICE
    )
    
    print(f"Loss: {CE_WEIGHT:.1f} × CE + {DICE_WEIGHT:.1f} × Dice")
    print(f"Class weights: {CLASS_WEIGHTS.tolist()}")
    
    # ========================================================================
    # Phase 1: Freeze Backbone, Train Head Only
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PHASE 1: FREEZE BACKBONE, TRAIN HEAD ONLY")
    print("=" * 80)
    print(f"Epochs: {PHASE1_EPOCHS}")
    print(f"Learning Rate: {LR_PHASE1}")
    
    # Freeze SegFormer backbone encoder
    for param in model.segformer.encoder.parameters():
        param.requires_grad = False
    
    # Create optimizer for head only
    head_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("segformer.encoder") and p.requires_grad
    ]
    optimizer_phase1 = optim.AdamW(head_params, lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)
    scheduler_phase1 = polynomial_decay_scheduler(optimizer_phase1, PHASE1_EPOCHS, LR_PHASE1)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_iou = 0.0
    best_model_path = os.path.join(output_dir, 'best_model_phase1.pth')
    
    for epoch in range(PHASE1_EPOCHS):
        print(f"\nPhase 1 - Epoch {epoch + 1}/{PHASE1_EPOCHS}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer_phase1, loss_fn, device, "Phase1-Train")
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Update scheduler
        scheduler_phase1.step()
        
        # Store history
        for key in history:
            if key.startswith('train'):
                history[key].append(train_metrics[key.replace('train_', '')])
            else:
                history[key].append(val_metrics[key.replace('val_', '')])
        
        # Print metrics
        print(f"  Train: Loss={train_metrics['loss']:.4f}, IoU={train_metrics['iou']:.4f}, Acc={train_metrics['acc']:.4f}")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, IoU={val_metrics['iou']:.4f}, Acc={val_metrics['acc']:.4f}")
        
        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Saved best model (IoU: {best_val_iou:.4f})")
    
    print(f"\nPhase 1 Complete! Best Val IoU: {best_val_iou:.4f}")
    
    # ========================================================================
    # Phase 2: Unfreeze All, Fine-tune End-to-End
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("PHASE 2: UNFREEZE ALL, FINE-TUNE END-TO-END")
    print("=" * 80)
    print(f"Epochs: {PHASE2_EPOCHS}")
    print(f"Learning Rate: {LR_PHASE2} (backbone), {LR_PHASE2 * 10} (head)")
    
    # Unfreeze SegFormer backbone encoder
    for param in model.segformer.encoder.parameters():
        param.requires_grad = True
    
    # Create optimizer with layer-wise LR
    optimizer_phase2 = create_optimizer_with_layer_wise_lr(model, LR_PHASE2, WEIGHT_DECAY)
    scheduler_phase2 = polynomial_decay_scheduler(optimizer_phase2, PHASE2_EPOCHS, LR_PHASE2)
    
    best_val_iou_phase2 = best_val_iou
    best_model_path_phase2 = os.path.join(output_dir, 'best_model_phase2.pth')
    
    for epoch in range(PHASE2_EPOCHS):
        print(f"\nPhase 2 - Epoch {epoch + 1}/{PHASE2_EPOCHS}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer_phase2, loss_fn, device, "Phase2-Train")
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        # Update scheduler
        scheduler_phase2.step()
        
        # Store history
        for key in history:
            if key.startswith('train'):
                history[key].append(train_metrics[key.replace('train_', '')])
            else:
                history[key].append(val_metrics[key.replace('val_', '')])
        
        # Print metrics
        print(f"  Train: Loss={train_metrics['loss']:.4f}, IoU={train_metrics['iou']:.4f}, Acc={train_metrics['acc']:.4f}")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, IoU={val_metrics['iou']:.4f}, Acc={val_metrics['acc']:.4f}")
        
        # Save best model
        if val_metrics['iou'] > best_val_iou_phase2:
            best_val_iou_phase2 = val_metrics['iou']
            torch.save(model.state_dict(), best_model_path_phase2)
            print(f"  → Saved best model (IoU: {best_val_iou_phase2:.4f})")
    
    print(f"\nPhase 2 Complete! Best Val IoU: {best_val_iou_phase2:.4f}")
    
    # ========================================================================
    # Save Final Model and Training History
    # ========================================================================
    
    final_model_path = os.path.join(output_dir, 'segformer_b2_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    print("\nGenerating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val', marker='s')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(PHASE1_EPOCHS, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
    
    # IoU
    axes[0, 1].plot(history['train_iou'], label='Train', marker='o')
    axes[0, 1].plot(history['val_iou'], label='Val', marker='s')
    axes[0, 1].set_title('Mean IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(PHASE1_EPOCHS, color='red', linestyle='--', alpha=0.5)
    
    # Accuracy
    axes[1, 0].plot(history['train_acc'], label='Train', marker='o')
    axes[1, 0].plot(history['val_acc'], label='Val', marker='s')
    axes[1, 0].set_title('Pixel Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(PHASE1_EPOCHS, color='red', linestyle='--', alpha=0.5)
    
    # Phase markers
    axes[1, 1].axis('off')
    info_text = f"""Training Summary:

Model: SegFormer-B2
Classes: {NUM_CLASSES}
Image Size: {IMG_SIZE}x{IMG_SIZE}
Batch Size: {BATCH_SIZE}

Phase 1 (Freeze Backbone): {PHASE1_EPOCHS} epochs
  - Learning Rate: {LR_PHASE1}
  - Best Val IoU: {best_val_iou:.4f}

Phase 2 (Fine-tune All): {PHASE2_EPOCHS} epochs
  - Backbone LR: {LR_PHASE2}
  - Head LR: {LR_PHASE2 * 10}
  - Best Val IoU: {best_val_iou_phase2:.4f}

Loss Function: {CE_WEIGHT:.1f}×CE + {DICE_WEIGHT:.1f}×Dice
Optimizer: AdamW (weight_decay={WEIGHT_DECAY})
Scheduler: Polynomial decay
Augmentation: ColorJitter, RandomCrop, Rotation, GaussianBlur, CutMix
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    print(f"Saved training curves to {os.path.join(output_dir, 'training_curves.png')}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best Val IoU (Phase 1): {best_val_iou:.4f}")
    print(f"Best Val IoU (Phase 2): {best_val_iou_phase2:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
