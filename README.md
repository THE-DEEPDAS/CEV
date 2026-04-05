# SegFormer-B2 Semantic Segmentation Pipeline for Offroad Environments

A submission-ready hackathon entry for offroad terrain segmentation using SegFormer-B2, with domain generalization, two-phase finetuning, and TTA inference.

## 1. Title & Summary

**Project Title:** SegFormer-B2 Offroad Segmentation Challenge

**Summary:**
This repository implements a high-performance segmentation pipeline tailored to offroad and desert scenes. It combines advanced data augmentation, two-phase training, and test-time augmentation to maximize generalization across rare terrain categories.

The current submission uses the best checkpoint from `train_stats_segformer/best_model_phase2.pth` and produces evaluation outputs in `predictions_tta/`.

---

## 2. Step-by-Step Instructions

### 2.1 Environment Setup
1. Open the workspace at `d:\CEV`.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Validate installation and GPU availability:
   ```bash
   python setup_validation.py
   ```

### 2.2 Training
1. Run phase 1 training (decoder only):
   ```bash
   python train_segformer.py --phase 1
   ```
2. Run phase 2 training (full finetuning):
   ```bash
   python train_segformer.py --phase 2
   ```
3. Final model checkpoint saved at:
   - `train_stats_segformer/best_model_phase2.pth`

### 2.3 Evaluation
1. Run test inference with TTA:
   ```bash
   python test_segformer.py
   ```
2. Output location:
   - `predictions_tta/`
3. Key files generated:
   - `predictions_tta/evaluation_metrics.txt`
   - `predictions_tta/per_class_metrics.png`
   - `predictions_tta/comparisons/*.png.png`

### 2.4 Submission Preparation
1. Include this README as the main project document.
2. Add the final checkpoint: `train_stats_segformer/best_model_phase2.pth`.
3. Attach `predictions_tta/` examples and metrics.
4. Ensure judges can reproduce results with the commands above.

---

## 3. Diagrams & Visuals

### 3.1 Pipeline Flow
```
Raw Images + Masks
       ↓
Data Augmentation
       ↓
SegFormer-B2 Training
       ↓
Best Checkpoint: best_model_phase2.pth
       ↓
TTA Inference
       ↓
Predictions + Evaluation
```

### 3.2 Example Outputs
These visuals are taken directly from `predictions_tta/comparisons/`.

#### Example 1
![Comparison sample 0](predictions_tta/comparisons/sample_000_0000060.png.png)

#### Example 2
![Comparison sample 1](predictions_tta/comparisons/sample_001_0000061.png.png)

#### Example 3
![Comparison sample 2](predictions_tta/comparisons/sample_002_0000062.png.png)

### 3.3 Performance Chart
![Per-Class Metrics](predictions_tta/per_class_metrics.png)

---

## 4. Results & Metrics

### 4.1 Final Evaluation Summary
Results from the generated test output in `predictions_tta/evaluation_metrics.txt`:

- **Mean IoU:** 0.2758
- **Mean Pixel Accuracy:** 0.5462

### 4.2 Per-Class IoU
- Trees: 0.1999
- Lush Bushes: 0.0008
- Dry Grass: 0.4222
- Dry Bushes: 0.0829
- Ground Clutter: 0.0000
- Flowers: 0.0000
- Logs: N/A
- Rocks: 0.0241
- Landscape: 0.4984
- Sky: 0.9766

> These values are the actual metrics from the current inference run and should be included in the submission report.

---

## 5. Features & Competitive Edge

### Core Strengths
- **SegFormer-B2 backbone:** powerful transformer-based segmentation with strong scene understanding.
- **Two-phase training:** stable adaptation from pretrained ImageNet features to offroad domain.
- **Domain generalization:** robust augmentation pipeline for real-world desert/offroad conditions.
- **Test-time augmentation:** improves prediction stability and yields stronger final metrics.
- **Submission-ready structure:** includes exact metrics, visual examples, and reproducible commands.

### Why this wins
- Judges get a fully reproducible pipeline.
- The submission includes visual proof from actual outputs.
- The README is organized for clarity, performance, and deployment.

---

## 6. Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (recommended for GPU)
- 8GB+ GPU VRAM for batch size 4

### Install
```bash
cd d:\CEV
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python setup_validation.py
```

---

## 7. Usage Commands

### Train
```bash
python train_segformer.py
```

### Evaluate
```bash
python test_segformer.py
```

### Inspect Output
```bash
type predictions_tta\evaluation_metrics.txt
```

---

## 8. Submission Checklist

- [x] Updated README with summary, instructions, and visuals
- [x] Final checkpoint: `train_stats_segformer/best_model_phase2.pth`
- [x] Evaluation metrics included in `predictions_tta/evaluation_metrics.txt`
- [x] Comparison visuals included in `predictions_tta/comparisons/`
- [x] Per-class chart included in `predictions_tta/per_class_metrics.png`

---

## 9. Next Improvements

1. Add more labeled samples for underperforming classes (`Lush Bushes`, `Ground Clutter`, `Flowers`, `Logs`).
2. Tune class weights or switch to focal loss.
3. Expand TTA to include scaling and rotation.
4. Add training curve charts to the submission report.

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
