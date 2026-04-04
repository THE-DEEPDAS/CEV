# SegFormer-B2 Semantic Segmentation Pipeline for Offroad Environments

A production-grade semantic segmentation pipeline optimized for desert/offroad environments using SegFormer-B2 with advanced domain generalization techniques.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate setup
python setup_validation.py

# 3. Train
python train_segformer.py

# 4. Evaluate
python test_segformer.py

# 5. Analyze results
python analyze_results.py
```

**Expected Performance:**
- Without TTA: 65-75% mIoU
- With TTA: 68-79% mIoU (+2-4% improvement)

---

## 📋 Features

### Model Architecture
- **SegFormer-B2** backbone with ImageNet pretrained weights
- Efficient transformer-based encoder with convolutional decoder
- Adapted for 9-class offroad segmentation

### Advanced Training Techniques

#### 1. **Class Remapping in Dataset**
Maps raw pixel values (0, 100, 200, ..., 10000) to consistent class IDs (0-8):
```python
100 → 0 (Trees)
200 → 1 (Lush Bushes)
...
10000 → 8 (Sky)
```

#### 2. **Domain Generalization Augmentation** (Critical!)
ColorJitter forces the model to learn **shape-based features** instead of color shortcuts:
- ColorJitter: brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
- RandomHorizontalFlip, RandomVerticalFlip
- RandomAffine: rotation ±10°, translation 10%, scale 0.9-1.1
- GaussianBlur: σ=0.1-2.0
- RandomGrayscale: p=0.1
- **CutMix**: Synthetically augments rare classes (Ground Clutter, Logs)

#### 3. **Weighted Loss Function**
Combines CrossEntropyLoss with class weights + Dice Loss:
```
total_loss = 0.7 × CrossEntropyLoss(weights) + 0.3 × DiceLoss
```

**Class Weights:**
- Rare classes (Ground Clutter, Logs): 8.0×
- Common classes: 1.0×
- Very common (Sky): 0.5×

#### 4. **AdamW Optimizer with Polynomial Decay**
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: Polynomial decay (power=1.0)
- **Layer-wise Learning Rates:**
  - Backbone (encoder): 1× lr = 6e-5
  - Head (decoder): 10× lr = 6e-4

#### 5. **Two-Phase Training**
Prevents catastrophic forgetting of ImageNet features:

**Phase 1: Freeze Backbone (10 epochs)**
- Frozen: Encoder (ImageNet features)
- Trainable: Decoder head
- Learning rate: 6e-5
- Purpose: Fast task-specific learning

**Phase 2: Finetune All (40-80 epochs)**
- Unfrozen: All layers
- Layer-wise LR: 1× backbone, 10× head
- Learning rate: 6e-5 (backbone), 6e-4 (head)
- Purpose: Adapt ImageNet features to domain

#### 6. **Test-Time Augmentation (TTA)**
At inference, runs 4 augmentations:
1. Original image
2. Horizontal flip
3. Vertical flip
4. Both flips (180°)

Averages softmax outputs then takes argmax:
```python
predictions = argmax(mean(softmax([original, h_flip, v_flip, both])))
```

**Benefit:** +2-4% mIoU with zero training overhead

---

## 📁 Project Structure

```
segformer_b2_pipeline/
├── train_segformer.py                  # Main training script
├── test_segformer.py                   # Inference with TTA
├── setup_validation.py                 # Setup checker
├── analyze_results.py                  # Results analyzer
│
├── QUICKSTART.md                       # 3-step guide
├── SEGFORMER_TRAINING_GUIDE.md        # Detailed documentation
├── IMPLEMENTATION_SUMMARY.md           # What was implemented
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── segformer/                          # Pretrained weights
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── preprocessor_config.json
│   └── README.md
│
├── train_stats_segformer/              # Training outputs
│   ├── best_model_phase1.pth
│   ├── best_model_phase2.pth          ← Best model
│   ├── segformer_b2_final.pth
│   ├── training_history.json
│   └── training_curves.png
│
└── predictions_tta/                    # Inference outputs
    ├── masks/                          # Raw predictions (0-8)
    ├── masks_color/                    # RGB visualizations
    ├── comparisons/                    # Ground truth comparisons
    ├── evaluation_metrics.txt
    ├── per_class_metrics.png
    └── inference_results.json
```

---

## 🔧 Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (recommended for GPU acceleration)
- 8GB+ GPU VRAM (for batch_size=4) or adjust batch size for CPU

### Setup

```bash
# Clone or navigate to project directory
cd offroad-segmentation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python setup_validation.py
```

---

## 📊 Dataset Format

Expected directory structure:

```
Offroad_Segmentation_Training_Dataset/
├── train/
│   ├── Color_Images/       (RGB images, e.g., 960×540)
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── Segmentation/       (Masks with values: 0, 100, 200, 300, 500, 550, 700, 800, 7100, 10000)
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
│
└── val/
    ├── Color_Images/
    └── Segmentation/

Offroad_Segmentation_testImages/
├── Color_Images/
└── Segmentation/

segformer/
├── config.json
├── pytorch_model.bin       ← Pretrained weights (required)
├── preprocessor_config.json
└── README.md
```

**Class Mapping:**
```
Raw Value  →  Class ID  →  Label
0          →  0         →  Background
100        →  0         →  Trees
200        →  1         →  Lush Bushes
300        →  2         →  Dry Grass
500        →  3         →  Dry Bushes
550        →  4         →  Ground Clutter (rare)
700        →  5         →  Logs (rare)
800        →  6         →  Rocks
7100       →  7         →  Landscape
10000      →  8         →  Sky
```

---

## 🎓 Usage Guide

### 1. Verify Setup

```bash
python setup_validation.py
```

Checks:
- ✓ CUDA availability
- ✓ Pretrained weights
- ✓ Dataset integrity
- ✓ Mask values

### 2. Train Model

```bash
python train_segformer.py
```

**What happens:**
- Phase 1 (10 epochs): Freezes backbone, trains head only
- Phase 2 (40 epochs): Unfreezes all, finetunes end-to-end
- Saves best model after each phase
- Generates training curves

**Expected time:**
- Phase 1: ~30 minutes (depends on GPU)
- Phase 2: ~2 hours
- Total: ~2.5 hours

**Outputs:**
```
train_stats_segformer/
├── best_model_phase1.pth
├── best_model_phase2.pth      ← USE THIS FOR INFERENCE
├── segformer_b2_final.pth
├── training_history.json
└── training_curves.png
```

### 3. Run Inference

```bash
python test_segformer.py
```

**Options:**
```bash
# Use custom model
python test_segformer.py --model_path custom_model.pth

# Custom data directory
python test_segformer.py --data_dir /path/to/test/data

# Save fewer comparison images (default 10)
python test_segformer.py --num_comparisons 5
```

**Outputs:**
```
predictions_tta/
├── masks/                    # Raw predictions (class IDs 0-8)
├── masks_color/              # RGB visualizations
├── comparisons/              # Side-by-side comparisons
├── evaluation_metrics.txt    # Per-class IoU
├── per_class_metrics.png     # Chart
└── inference_results.json    # JSON results
```

### 4. Analyze Results

```bash
python analyze_results.py
```

**Output:**
- Phase-wise convergence analysis
- Per-class performance breakdown
- Overfitting/underfitting detection
- Recommendations

```bash
# Compare multiple models
python analyze_results.py --compare model_v1:path1 model_v2:path2
```

---

## 🔧 Configuration

### Training Hyperparameters

Edit `train_segformer.py`:

```python
# Data
BATCH_SIZE = 4              # Reduce to 2 if GPU OOM
IMG_SIZE = 512              # Keep for detail; reduce to 384 if needed

# Training phases
PHASE1_EPOCHS = 10          # Increase to 20 if head not converging
PHASE2_EPOCHS = 40          # Increase to 80 if still improving
LR_PHASE1 = 6e-5            # Standard for SegFormer
LR_PHASE2 = 6e-5            # Will use layer-wise (1× backbone, 10× head)
WEIGHT_DECAY = 0.01         # L2 regularization

# Loss weights
CE_WEIGHT = 0.7             # CrossEntropy weight
DICE_WEIGHT = 0.3           # Dice weight

# Class weights (adjust based on performance)
CLASS_WEIGHTS = [
    1.0,    # 0: Background
    1.0,    # 1: Trees
    1.0,    # 2: Lush Bushes
    1.0,    # 3: Dry Grass
    1.0,    # 4: Dry Bushes
    8.0,    # 5: Ground Clutter (rare) ← increase if underfitting
    8.0,    # 6: Logs (rare) ← increase if underfitting
    1.0,    # 7: Rocks
    0.5,    # 8: Sky (very common) ← decrease if overfitting
]
```

### Data Augmentation

The training script includes optimized augmentation pipeline. Modify in `get_train_transforms()`:

```python
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)  # Critical!
RandomGrayscale(p=0.1)
GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
# ... etc
```

**Note:** ColorJitter is the #1 factor for desert generalization.

---

## 📈 Expected Results

### Phase-wise Performance

| Phase | Epochs | Focus | Val IoU | Time |
|-------|--------|-------|---------|------|
| 1 | 10 | Head only | 50-60% | 30 min |
| 2 | 40 | Full model | 65-75% | 2 hours |
| +TTA | - | Inference | 68-79% | N/A |

### Per-class Performance (typical)

```
Background: 0.85 ✓ (common, easy)
Trees: 0.72 ✓ (distinctive shape)
Dry Grass: 0.68 ✓ (distinctive color)
Sky: 0.88 ✓ (very common, easy)
Ground Clutter: 0.45 ⚠ (rare, varied appearance)
Logs: 0.52 ⚠ (rare, small patches)
```

*Actual results depend on dataset quality and size.*

---

## 🐛 Troubleshooting

### Problem: GPU Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# In train_segformer.py
BATCH_SIZE = 2          # or 1
IMG_SIZE = 384          # instead of 512
```

Or use CPU (much slower):
```bash
# Model will automatically use CPU if CUDA unavailable
```

### Problem: Training Loss Not Decreasing

**Causes:**
- Learning rate too high
- Learning rate too low
- Dataset issues (check with setup_validation.py)
- Batch size too small

**Solutions:**
```python
# Try lower LR
LR_PHASE2 = 3e-5

# Or increase epochs
PHASE2_EPOCHS = 80

# Or increase batch size (if GPU memory allows)
BATCH_SIZE = 8
```

### Problem: Specific Classes Have Very Low IoU

**Example:** Ground Clutter always < 0.3

**Solutions:**
1. **Increase class weight:**
   ```python
   CLASS_WEIGHTS[4] = 12.0  # or higher
   ```

2. **Increase CutMix:**
   - CutMix already enabled with p=0.3
   - Creates synthetic samples of rare classes

3. **Train longer:**
   ```python
   PHASE2_EPOCHS = 80  # or more
   ```

4. **Check data quality:**
   ```bash
   python setup_validation.py
   ```
   - Are Ground Clutter masks clearly labeled?
   - Are there enough samples?

### Problem: Model Overfitting

**Sign:** Train IoU >> Val IoU (e.g., 0.85 vs 0.65)

**Solutions:**
```python
# Increase augmentation
# Already quite strong; could increase:
# - ColorJitter brightness/contrast/saturation
# - RandomGrayscale probability
# - CutMix probability

# Or reduce weight of common classes
CLASS_WEIGHTS[8] = 0.2  # Reduce Sky weight

# Or increase regularization
WEIGHT_DECAY = 0.05  # Increase from 0.01
```

### Problem: CUDA Not Available

**Error:** `CUDA not available - will use CPU (very slow)`

**Check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions:**
- Install CUDA-compatible PyTorch: [pytorch.org](https://pytorch.org)
- Or use CPU (very slow, 10-50x slower)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](QUICKSTART.md) | 3-step guide to get started |
| [SEGFORMER_TRAINING_GUIDE.md](SEGFORMER_TRAINING_GUIDE.md) | Detailed technical guide |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | What was implemented and why |

---

## 🎯 Key Takeaways

### Why ColorJitter is Critical
Desert environments all look similar (browns, tans). Without ColorJitter:
```
❌ Model: Tree = green pixels → fails in different desert
✓ Model: Tree = shape + boundaries → works everywhere
```

### Why Two-Phase Training
Direct fine-tuning destroys ImageNet features. Two-phase approach:
```
Phase 1: Learn task quickly (head)
Phase 2: Adapt ImageNet features gently (backbone at 1× LR)
```

### Why TTA Works
Different crops/flips provide different views:
```
Image 1: Sees top-left region
Image 2: Sees top-right region
Image 3: Sees bottom-left region
Image 4: Sees bottom-right region
Average: More confident prediction
```

### Why Class Weights Matter
Without weights, model ignores rare classes:
```
❌ Without weights: Predicts Sky for everything (80% accuracy!)
✓ With weights: Rare classes: 8.0×, Sky: 0.5×
```

---

## 📖 References

- **SegFormer**: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.03722)
- **Transformers**: [HuggingFace Transformers Library](https://huggingface.co/transformers)
- **Domain Generalization**: [CVPR/ICCV Papers on DG](https://arxiv.org/)

---

## 📝 Citation

If you use this code in your research or project:

```bibtex
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.03722},
  year={2021}
}
```

---

## 📞 Support

For issues or questions:

1. **Check logs**: Review console output for error messages
2. **Validate**: Run `python setup_validation.py`
3. **Analyze**: Run `python analyze_results.py`
4. **Read**: Check documentation files

---

## 📄 License

[Specify your license here]

---

## 🎉 Getting Started

Ready to train? Run these three commands:

```bash
# 1. Verify setup
python setup_validation.py

# 2. Start training
python train_segformer.py

# 3. Run inference
python test_segformer.py
```

**Happy training! 🚀**

For detailed guide, see [QUICKSTART.md](QUICKSTART.md).
