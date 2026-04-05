#!/usr/bin/env python
"""
Setup validation and preparation script for SegFormer-B2 training.
Checks data integrity, dataset structure, and pretrained weights.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch

def check_cuda():
    """Check CUDA availability."""
    print("=" * 80)
    print("CHECKING CUDA")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("✗ CUDA not available - will use CPU (very slow)")
    print()


def check_segformer_weights(script_dir):
    """Check SegFormer pretrained weights."""
    print("=" * 80)
    print("CHECKING SEGFORMER PRETRAINED WEIGHTS")
    print("=" * 80)
    
    segformer_dir = os.path.join(script_dir, 'segformer')
    
    if not os.path.exists(segformer_dir):
        print(f"✗ segformer/ directory not found at {segformer_dir}")
        return False
    
    required_files = ['config.json', 'pytorch_model.bin', 'preprocessor_config.json']
    all_exist = True
    
    for filename in required_files:
        filepath = os.path.join(segformer_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {filename} NOT FOUND")
            all_exist = False
    
    if all_exist:
        try:
            with open(os.path.join(segformer_dir, 'config.json')) as f:
                config = json.load(f)
                print(f"  Model architecture: {config.get('architectures', ['Unknown'])[0]}")
                print(f"  Num labels in config: {config.get('num_labels', config.get('num_labels', 'Unknown'))}")
        except Exception as e:
            print(f"  Warning: Could not read config: {e}")
    
    print()
    return all_exist


def check_dataset(data_dir, split_name):
    """Check dataset structure and content."""
    print(f"CHECKING {split_name.upper()} DATASET")
    print("-" * 80)
    
    image_dir = os.path.join(data_dir, 'Color_Images')
    mask_dir = os.path.join(data_dir, 'Segmentation')
    
    if not os.path.exists(image_dir):
        print(f"✗ {split_name}/Color_Images/ not found")
        return False
    
    if not os.path.exists(mask_dir):
        print(f"✗ {split_name}/Segmentation/ not found")
        return False
    
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"✓ Color_Images: {len(image_files)} images")
    print(f"✓ Segmentation: {len(mask_files)} masks")
    
    if len(image_files) == 0 or len(mask_files) == 0:
        print("✗ No images or masks found!")
        return False
    
    if len(image_files) != len(mask_files):
        print(f"✗ Mismatch: {len(image_files)} images vs {len(mask_files)} masks")
    
    # Check first image-mask pair
    try:
        img_path = os.path.join(image_dir, image_files[0])
        mask_path = os.path.join(mask_dir, image_files[0])
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        print(f"✓ Sample image: {img.size} {img.mode}")
        print(f"✓ Sample mask: {mask.size} {mask.mode}")
        
        # Check mask values
        mask_array = np.array(mask)
        unique_values = np.unique(mask_array)
        print(f"✓ Mask unique values: {sorted(unique_values.tolist())}")
        
        # Check if values match expected class mapping
        expected_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 28, 39, 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000}
        if set(unique_values) <= expected_values:
            print(f"✓ Mask values match expected class mapping")
        else:
            unknown = set(unique_values) - expected_values
            print(f"⚠ Unknown mask values: {unknown}")
        
    except Exception as e:
        print(f"✗ Error checking sample: {e}")
        return False
    
    print()
    return True


def check_datasets(script_dir):
    """Check all datasets."""
    print("=" * 80)
    print("CHECKING DATASETS")
    print("=" * 80)
    print()
    
    all_ok = True
    
    # Training dataset
    train_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    if check_dataset(train_dir, 'train'):
        print()
    else:
        print("✗ Training dataset check failed\n")
        all_ok = False
    
    # Validation dataset
    val_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')
    if check_dataset(val_dir, 'validation'):
        print()
    else:
        print("⚠ Validation dataset check failed (may not exist yet)\n")
    
    # Test dataset
    test_dir = os.path.join(script_dir, 'Offroad_Segmentation_testImages')
    if check_dataset(test_dir, 'test'):
        print()
    else:
        print("⚠ Test dataset check failed (may not exist yet)\n")
    
    return all_ok


def print_configuration():
    """Print recommended configuration."""
    print("=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)
    print("""
SegFormer-B2 Training Setup:
  Model: SegFormer-B2 (pretrained on ADE20K)
  Num Classes: 9
  Input Size: 512×512
  Batch Size: 4 (adjust based on GPU memory)
  
Phase 1 (Freeze Backbone):
  Epochs: 10
  Learning Rate: 6e-5
  
Phase 2 (Fine-tune All):
  Epochs: 40-80
  Backbone LR: 6e-5 (1×)
  Head LR: 6e-5 × 10 = 6e-4 (10×)
  
Loss Function:
  Total = 0.7 × CrossEntropyLoss(weights) + 0.3 × DiceLoss
  
Data Augmentation (Critical!):
  - ColorJitter: brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
  - RandomHorizontalFlip: p=0.5
  - RandomVerticalFlip: p=0.2
  - GaussianBlur: σ=0.1-2.0
  - RandomGrayscale: p=0.1
  - RandomAffine: rotation±10°, translate 10%, scale 0.9-1.1
  - CutMix (rare classes): p=0.3
  
Inference:
  - Test-Time Augmentation (TTA) enabled
  - 4 augmentations (original + 3 flips)
  - Average softmax + argmax
  - Expected +2-4% mIoU improvement

Class Weights:
  Background: 1.0
  Trees: 1.0
  Lush Bushes: 1.0
  Dry Grass: 1.0
  Dry Bushes: 1.0
  Ground Clutter (rare): 8.0×
  Logs (rare): 8.0×
  Rocks: 1.0
  Sky (common): 0.5×
""")


def print_usage():
    """Print usage instructions."""
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Review the configuration above and adjust if needed

2. Start training:
   python train_segformer.py

3. Monitor progress:
   - Training metrics printed each epoch
   - Curves saved to train_stats_segformer/training_curves.png
   - Best models saved to train_stats_segformer/

4. Run inference with TTA:
   python test_segformer.py

5. View results:
   - Predictions saved to predictions_tta/
   - Metrics in predictions_tta/evaluation_metrics.txt
   - Charts in predictions_tta/per_class_metrics.png

For detailed documentation:
   Read SEGFORMER_TRAINING_GUIDE.md
""")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "SegFormer-B2 Training Pipeline - Setup Validation".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Run checks
    check_cuda()
    weights_ok = check_segformer_weights(script_dir)
    datasets_ok = check_datasets(script_dir)
    
    # Print results
    print("=" * 80)
    print("SETUP STATUS")
    print("=" * 80)
    
    if weights_ok and datasets_ok:
        print("✓ All checks passed! Ready to train.")
    else:
        print("✗ Some checks failed. See details above.")
        if not weights_ok:
            print("\n  Fix: Download SegFormer-B2 weights to segformer/ folder")
        if not datasets_ok:
            print("\n  Fix: Ensure training dataset is in Offroad_Segmentation_Training_Dataset/")
    
    print()
    
    print_configuration()
    print()
    print_usage()
    print()


if __name__ == "__main__":
    main()

