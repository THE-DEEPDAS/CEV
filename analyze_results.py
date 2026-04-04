"""
Analysis and comparison utilities for SegFormer-B2 training.
Analyzes training metrics, compares models, and generates reports.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse


def load_training_history(history_path):
    """Load training history from JSON file."""
    with open(history_path, 'r') as f:
        return json.load(f)


def analyze_training(stats_dir):
    """Analyze training statistics."""
    print("=" * 80)
    print("TRAINING ANALYSIS")
    print("=" * 80)
    print()
    
    history_path = os.path.join(stats_dir, 'training_history.json')
    
    if not os.path.exists(history_path):
        print(f"✗ Training history not found at {history_path}")
        return
    
    history = load_training_history(history_path)
    
    # Phase 1 analysis (first 10 epochs)
    print("PHASE 1: FREEZE BACKBONE (Epochs 1-10)")
    print("-" * 80)
    
    phase1_train_loss = history['train_loss'][:10]
    phase1_val_loss = history['val_loss'][:10]
    phase1_train_iou = history['train_iou'][:10]
    phase1_val_iou = history['val_iou'][:10]
    
    print(f"Train Loss:      {phase1_train_loss[0]:.4f} → {phase1_train_loss[-1]:.4f} ({(phase1_train_loss[-1]/phase1_train_loss[0] - 1)*100:+.1f}%)")
    print(f"Val Loss:        {phase1_val_loss[0]:.4f} → {phase1_val_loss[-1]:.4f} ({(phase1_val_loss[-1]/phase1_val_loss[0] - 1)*100:+.1f}%)")
    print(f"Train IoU:       {phase1_train_iou[0]:.4f} → {phase1_train_iou[-1]:.4f} ({(phase1_train_iou[-1]/phase1_train_iou[0] - 1)*100:+.1f}%)")
    print(f"Val IoU:         {phase1_val_iou[0]:.4f} → {phase1_val_iou[-1]:.4f} ({(phase1_val_iou[-1]/phase1_val_iou[0] - 1)*100:+.1f}%)")
    print(f"Best Val IoU:    {max(phase1_val_iou):.4f} (Epoch {np.argmax(phase1_val_iou) + 1})")
    print()
    
    # Phase 2 analysis
    print("PHASE 2: FINE-TUNE ALL (Epochs 11+)")
    print("-" * 80)
    
    phase2_train_loss = history['train_loss'][10:]
    phase2_val_loss = history['val_loss'][10:]
    phase2_train_iou = history['train_iou'][10:]
    phase2_val_iou = history['val_iou'][10:]
    
    if len(phase2_train_loss) > 0:
        print(f"Train Loss:      {phase2_train_loss[0]:.4f} → {phase2_train_loss[-1]:.4f} ({(phase2_train_loss[-1]/phase2_train_loss[0] - 1)*100:+.1f}%)")
        print(f"Val Loss:        {phase2_val_loss[0]:.4f} → {phase2_val_loss[-1]:.4f} ({(phase2_val_loss[-1]/phase2_val_loss[0] - 1)*100:+.1f}%)")
        print(f"Train IoU:       {phase2_train_iou[0]:.4f} → {phase2_train_iou[-1]:.4f} ({(phase2_train_iou[-1]/phase2_train_iou[0] - 1)*100:+.1f}%)")
        print(f"Val IoU:         {phase2_val_iou[0]:.4f} → {phase2_val_iou[-1]:.4f} ({(phase2_val_iou[-1]/phase2_val_iou[0] - 1)*100:+.1f}%)")
        print(f"Best Val IoU:    {max(phase2_val_iou):.4f} (Epoch {10 + np.argmax(phase2_val_iou) + 1})")
        print()
    
    # Overall statistics
    print("OVERALL")
    print("-" * 80)
    print(f"Total Epochs:    {len(history['train_loss'])}")
    print(f"Best Val IoU:    {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})")
    print(f"Final Val IoU:   {history['val_iou'][-1]:.4f}")
    print(f"Final Val Loss:  {history['val_loss'][-1]:.4f}")
    print(f"Final Train Acc: {history['train_pixel_acc'][-1]:.4f}")
    print(f"Final Val Acc:   {history['val_pixel_acc'][-1]:.4f}")
    print()
    
    # Check for overfitting/underfitting
    print("CONVERGENCE ANALYSIS")
    print("-" * 80)
    
    avg_train_loss = np.mean(history['train_loss'])
    avg_val_loss = np.mean(history['val_loss'])
    final_train_iou = history['train_iou'][-1]
    final_val_iou = history['val_iou'][-1]
    
    if final_train_iou - final_val_iou > 0.10:
        print("⚠ Overfitting detected (gap > 10%)")
        print("   Recommendation: Increase data augmentation or class weight for Sky")
    elif final_val_iou - final_train_iou > 0.05:
        print("⚠ Possible underfitting")
        print("   Recommendation: Increase training epochs or decrease weight decay")
    else:
        print("✓ Good convergence (gap < 10%)")
    
    if avg_val_loss > avg_train_loss * 1.5:
        print("⚠ High validation loss relative to training")
        print("   Recommendation: Check dataset distribution, increase augmentation")
    else:
        print("✓ Training and validation loss reasonably aligned")
    
    print()


def load_inference_results(results_path):
    """Load inference results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_inference(results_dir):
    """Analyze inference results."""
    print("=" * 80)
    print("INFERENCE ANALYSIS")
    print("=" * 80)
    print()
    
    results_path = os.path.join(results_dir, 'inference_results.json')
    
    if not os.path.exists(results_path):
        print(f"✗ Inference results not found at {results_path}")
        return
    
    results = load_inference_results(results_path)
    
    CLASS_NAMES = [
        'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
        'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
    ]
    
    print(f"Mean IoU:          {results['mean_iou']:.4f}")
    print(f"Mean Pixel Acc:    {results['mean_pixel_acc']:.4f}")
    print()
    
    print("PER-CLASS IoU:")
    print("-" * 80)
    
    class_ious = results['class_iou']
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, class_ious)):
        if np.isnan(iou):
            iou_str = "N/A (no samples)"
        else:
            iou_str = f"{iou:.4f}"
            if iou < 0.3:
                iou_str += " ✗ (very low)"
            elif iou < 0.5:
                iou_str += " ⚠ (low)"
            elif iou > 0.75:
                iou_str += " ✓ (excellent)"
        
        print(f"  {i}: {name:<20} : {iou_str}")
    
    print()
    
    # Find problematic classes
    valid_ious = [iou for iou in class_ious if not np.isnan(iou)]
    if valid_ious:
        worst_idx = np.nanargmin(class_ious)
        best_idx = np.nanargmax(class_ious)
        
        print("PERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"Best:   {CLASS_NAMES[best_idx]:<20} ({class_ious[best_idx]:.4f})")
        print(f"Worst:  {CLASS_NAMES[worst_idx]:<20} ({class_ious[worst_idx]:.4f})")
        
        if class_ious[worst_idx] < 0.3:
            print(f"\n⚠ {CLASS_NAMES[worst_idx]} has very low IoU. Recommendations:")
            print(f"  - Check if class is underrepresented in training data")
            print(f"  - Increase class weight (currently likely 1.0-8.0)")
            print(f"  - Increase CutMix probability for rare classes")
            print(f"  - Try increasing Phase 2 epochs")
    
    print()


def generate_comparison_report(results_dirs):
    """Generate comparison report for multiple results."""
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print()
    
    models = {}
    for name, path in results_dirs:
        results_path = os.path.join(path, 'inference_results.json')
        if os.path.exists(results_path):
            results = load_inference_results(results_path)
            models[name] = results['mean_iou']
    
    if not models:
        print("No results found to compare")
        return
    
    # Sort by performance
    sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
    
    print("Model Rankings (by Mean IoU):")
    print("-" * 80)
    for rank, (name, iou) in enumerate(sorted_models, 1):
        print(f"{rank}. {name:<30} : {iou:.4f}")
    
    # Calculate differences
    if len(sorted_models) > 1:
        best_iou = sorted_models[0][1]
        print()
        print("Performance Difference vs Best:")
        print("-" * 80)
        for name, iou in sorted_models[1:]:
            diff = (iou - best_iou) * 100
            print(f"  {name:<30} : {diff:+.2f}%")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze SegFormer-B2 training and inference results')
    parser.add_argument('--training_dir', type=str, default='train_stats_segformer',
                        help='Directory with training statistics')
    parser.add_argument('--inference_dir', type=str, default='predictions_tta',
                        help='Directory with inference results')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare multiple result directories (format: name1:path1 name2:path2 ...)')
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "SegFormer-B2 Training & Inference Analysis".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Analyze training
    if os.path.exists(args.training_dir):
        analyze_training(args.training_dir)
    else:
        print(f"⚠ Training directory not found: {args.training_dir}\n")
    
    # Analyze inference
    if os.path.exists(args.inference_dir):
        analyze_inference(args.inference_dir)
    else:
        print(f"⚠ Inference directory not found: {args.inference_dir}\n")
    
    # Compare models if requested
    if args.compare:
        results_dirs = []
        for item in args.compare:
            name, path = item.split(':')
            results_dirs.append((name, path))
        generate_comparison_report(results_dirs)
    
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
