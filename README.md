# Final Submission Document

## 1. Title & Summary

**Purpose:** This is the final hackathon submission document for the offroad segmentation project.

**What it does:**
- trains a SegFormer-B2 model for offroad terrain segmentation
- uses two-phase training and test-time augmentation
- generates evaluation metrics and comparison visuals

**Final model checkpoint:** available from Kaggle at https://www.kaggle.com/models/deepdas07/offroad-segmentation

**Final approach & validation:** documented in `cevass.ipynb`, including three new final validation figures.

---

## 2. Step-by-Step Instructions

### 2.1 Setup
1. Open the project folder at `d:\CEV`.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Validate the install:
   ```bash
   python setup_validation.py
   ```

### 2.2 Training
1. Train phase 1:
   ```bash
   python train_segformer.py --phase 1
   ```
2. Train phase 2:
   ```bash
   python train_segformer.py --phase 2
   ```
3. Final checkpoint:
   - `train_stats_segformer/best_model_phase2.pth`

### 2.3 Final Validation & Figures
The final validation approach is documented in `cevass.ipynb` and includes three new validation figures produced on held-out data.

- Model design:
  - Two-phase SegFormer-B2 training
  - Class remapping for offroad terrain labels
  - Strong augmentation for domain generalization
  - Final test-time augmentation (TTA) for robust predictions

- Validation process:
  - Validate on the held-out `val/` dataset
  - Record per-class IoU and pixel accuracy
  - Compare input, ground truth, and prediction side-by-side

![Validation comparison 1](predictions_tta/comparisons/sample_000_0000060.png.png)

![Validation comparison 2](predictions_tta/comparisons/sample_001_0000061.png.png)

![Per-class metrics](predictions_tta/per_class_metrics.png)

### 2.4 Evaluation
1. Download the final checkpoint from Kaggle and save it locally as `best_model.pt`.
2. Run inference with TTA:
   ```bash
   python test_segformer.py --model_path best_model.pt
   ```
3. Review outputs in `predictions_tta/`.
4. Main result files:
   - `predictions_tta/evaluation_metrics.txt`
   - `predictions_tta/per_class_metrics.png`
   - `predictions_tta/comparisons/`

### 2.5 Submission
1. Use this README as the final document.
2. Include the final checkpoint and `predictions_tta/`.
3. Confirm exact reproduction with the commands above.
4. Final validation analysis and three new validation figures are captured in `cevass.ipynb`.

---

## 3. Diagrams & Visuals

### 3.1 Pipeline Flow
```
Dataset → Augmentation → SegFormer-B2 Training → best_model_phase2.pth → TTA Inference → Predictions
```

### 3.2 Example Outputs
These visuals come from `predictions_tta/comparisons/`.

#### Example 1
![Comparison sample 0](predictions_tta/comparisons/sample_000_0000060.png.png)

#### Example 2
![Comparison sample 1](predictions_tta/comparisons/sample_001_0000061.png.png)

#### Example 3
![Comparison sample 2](predictions_tta/comparisons/sample_002_0000062.png.png)

### 3.3 Performance Chart
![Per-Class Metrics](predictions_tta/per_class_metrics.png)

---

## 4. Graphs & Charts

### 4.1 Final Metrics
From `predictions_tta/evaluation_metrics.txt`:
- Mean IoU: **0.2880**
- Pixel Accuracy: **0.5796**

### 4.2 Per-Class IoU
- Trees: 0.1755
- Lush Bushes: 0.0011
- Dry Grass: 0.4263
- Dry Bushes: 0.1608
- Ground Clutter: 0.0000
- Flowers: 0.0000
- Logs: 0.0000
- Rocks: 0.0301
- Landscape: 0.5479
- Sky: 0.9667

### 4.3 Example Entry
- Task: Model training on dataset
- Initial IoU Score: 0.2758
- Issue Faced: low recall for rare classes
- Solution: stronger augmentation and TTA

---

## 5. Documented Steps

- Dataset mapping: raw masks are remapped to 9 classes.
- Training: phase 1 freezes the backbone, phase 2 finetunes the whole model.
- Evaluation: `test_segformer.py` creates prediction comparisons and metrics.

---

## 6. Failure Cases and Solutions

### Failure 1: Rare class recall
- Issue: Logs and Flowers are missing or weak.
- Evidence: IoU values are 0 or N/A.
- Fix: add rare-class examples and increase class weights.

### Failure 2: Confusing terrain types
- Issue: Dry Bushes appear as Dry Grass.
- Evidence: comparison images show mixed textures.
- Fix: add more texture variety and occlusion augmentation.

### Before & After Examples
- `predictions_tta/comparisons/sample_000_0000060.png.png`
- `predictions_tta/comparisons/sample_001_0000061.png.png`
- `predictions_tta/comparisons/sample_002_0000062.png.png`

---

## 7. Notes for Judges

- This README is the single final submission document.
- It includes exact results and visual proof from the current output.
- The final model and outputs are ready for evaluation.

