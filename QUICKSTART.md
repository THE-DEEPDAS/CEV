# SegFormer-B2 Quick Start Guide

## Overview

This is a production-ready segmentation pipeline using:
- **SegFormer-B2** backbone with ImageNet pretrained weights
- **Advanced domain generalization** techniques for desert environments
- **Two-phase training** for optimal convergence
- **Test-Time Augmentation (TTA)** for +2-4% mIoU improvement

## Installation

1. **Install dependencies:**
```bash
pip install torch torchvision transformers opencv-python tqdm numpy pillow matplotlib
```

2. **Verify setup:**
```bash
python setup_validation.py
```

This will check:
- ✓ CUDA availability
- ✓ Pretrained weights in `segformer/` folder
- ✓ Dataset structure and integrity
- ✓ All required files

## Quick Start: 3 Steps

### Step 1: Prepare Data
```
Offroad_Segmentation_Training_Dataset/
├── train/
│   ├── Color_Images/     (RGB images)
│   └── Segmentation/     (masks with values 0, 100, 200, ..., 10000)
└── val/
    ├── Color_Images/
    └── Segmentation/

Offroad_Segmentation_testImages/
├── Color_Images/
└── Segmentation/
```

### Step 2: Train
```bash
python train_segformer.py
```

**Output:**
- `train_stats_segformer/best_model_phase2.pth` ← Best model
- `train_stats_segformer/training_curves.png` ← Training metrics

**Training time:**
- Phase 1 (freeze backbone): ~30 minutes (10 epochs)
- Phase 2 (finetune all): ~2-3 hours (40 epochs)
- *Times depend on GPU; CPU will be much slower*

### Step 3: Evaluate
```bash
python test_segformer.py
```

**Output:**
- `predictions_tta/masks/` ← Predicted masks
- `predictions_tta/masks_color/` ← Colored visualizations
- `predictions_tta/comparisons/` ← Side-by-side comparisons
- `predictions_tta/evaluation_metrics.txt` ← Quantitative results

## Expected Results

| Metric | Phase 1 | Phase 2 | Phase 2 + TTA |
|--------|---------|---------|---------------|
| Val IoU | 50-60% | 65-75% | 68-79% |
| Training time | 10 min | 2 hours | N/A |

*Results depend on dataset size/quality; adjust hyperparameters as needed*

## Key Configuration

Edit `train_segformer.py` to customize:

```python
# Data
BATCH_SIZE = 4          # Reduce if GPU OOM
IMG_SIZE = 512          # Keep at 512 for detail

# Training
PHASE1_EPOCHS = 10      # Increase if head not converging
PHASE2_EPOCHS = 40      # Increase if still improving

# Loss (usually good defaults)
CE_WEIGHT = 0.7         # Cross-entropy loss weight
DICE_WEIGHT = 0.3       # Dice loss weight

# Class weights (adjust if specific classes underperform)
CLASS_WEIGHTS = [
    1.0,    # Background
    1.0,    # Trees
    1.0,    # Lush Bushes
    1.0,    # Dry Grass
    1.0,    # Dry Bushes
    8.0,    # Ground Clutter (rare) ← increase if underfitting
    8.0,    # Logs (rare) ← increase if underfitting
    1.0,    # Rocks
    0.5,    # Sky (very common) ← decrease if overfitting
]
```

## Troubleshooting

### Problem: GPU Out of Memory
**Solution:**
```python
BATCH_SIZE = 2  # or even 1
```

### Problem: Class X has very low IoU
**Solution 1:** Increase class weight
```python
CLASS_WEIGHTS[X] = 10.0  # or higher
```

**Solution 2:** Verify data quality
- Check if class samples are clear and labeled correctly
- Run `python setup_validation.py` to inspect masks

**Solution 3:** More training
```python
PHASE2_EPOCHS = 80  # or higher
```

### Problem: High training loss, low validation loss
**Solution:** Your model might be too simple or learning rate too high
```python
LR_PHASE2 = 3e-5  # Lower learning rate
PHASE2_EPOCHS = 60  # More epochs
```

### Problem: Model not improving after phase 1
**Solution:** Phase 1 head-only training needs more epochs
```python
PHASE1_EPOCHS = 20  # or more
```

## Advanced Usage

### Custom Model Checkpoint
```bash
python test_segformer.py --model_path path/to/your/model.pth
```

### Inference Without TTA (faster)
```bash
python test_segformer.py --use_tta
```

### Analyze Results
```bash
python analyze_results.py \
  --training_dir train_stats_segformer \
  --inference_dir predictions_tta
```

### Compare Multiple Models
```bash
python analyze_results.py \
  --compare \
    model_v1:predictions_tta_v1 \
    model_v2:predictions_tta_v2 \
    model_v3:predictions_tta_v3
```

## Performance Tips

### #1: ColorJitter is Critical
Without strong color augmentation, model memorizes desert colors. This severely limits generalization to different environments.

```python
# In train_segformer.py - this is already optimized
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
```

### #2: Two-Phase Training
Never skip this. It prevents forgetting ImageNet features while adapting to your task.

### #3: Use TTA at Inference
+2-4% mIoU for free with zero training overhead.

### #4: CutMix for Rare Classes
Synthetically creates more training samples of rare classes automatically.

### #5: Weighted Loss
Ensures rare classes (Ground Clutter, Logs) get sufficient attention.

## File Structure After Training

```
segformer_b2_final.pth
├── train_stats_segformer/              # Training outputs
│   ├── best_model_phase1.pth           # Best during phase 1
│   ├── best_model_phase2.pth           # Best overall (use for inference)
│   ├── segformer_b2_final.pth          # Final after all epochs
│   ├── training_history.json           # Raw metrics
│   └── training_curves.png             # Visual curves
│
├── predictions_tta/                    # Inference outputs
│   ├── masks/                          # Raw class IDs (0-8)
│   ├── masks_color/                    # RGB visualizations
│   ├── comparisons/                    # Comparison images
│   ├── evaluation_metrics.txt          # Per-class IoU
│   ├── per_class_metrics.png           # Bar chart
│   └── inference_results.json          # Raw results
```

## Understanding the Output

### Predicted Mask Values (0-8)
```
0 = Background
1 = Trees
2 = Lush Bushes
3 = Dry Grass
4 = Dry Bushes
5 = Ground Clutter
6 = Logs
7 = Rocks
8 = Landscape
```

### Metrics
- **IoU (Intersection over Union)**: Core segmentation metric (0-1, higher is better)
  - IoU = (True Positives) / (True Positives + False Positives + False Negatives)
  - Per-class IoU shows performance for each class
  - Mean IoU averages across all classes

- **Pixel Accuracy**: Percentage of correctly classified pixels (0-1)
  - Less sensitive to class imbalance than IoU

## Reference

- **SegFormer Paper**: [arxiv.org/abs/2105.03722](https://arxiv.org/abs/2105.03722)
- **Transformers Library**: [huggingface.co/transformers](https://huggingface.co/transformers)
- **Domain Generalization**: Techniques from CVPR/ICCV papers

## Citation

If you use this code in research, please cite:

```bibtex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.03722},
  year={2021}
}
```

## Support

For issues or questions:

1. **Check logs**: Review console output for error messages
2. **Validate setup**: Run `python setup_validation.py` to check configuration
3. **Analyze results**: Run `python analyze_results.py` to understand convergence
4. **Check documentation**: Read `SEGFORMER_TRAINING_GUIDE.md` for detailed explanations

## License

[Specify your license here]

---

**Happy training! 🚀**
