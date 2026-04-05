# SegFormer-B2 Offroad Segmentation: Complete Implementation Guide

## a. Title & Summary

**Purpose:** This document provides a comprehensive, structured guide for setting up, training, testing, and evaluating the SegFormer-B2 model for offroad terrain segmentation. The guide covers the entire pipeline from environment setup to model deployment, including advanced techniques like two-phase training, data augmentation, and test-time augmentation (TTA). The goal is to enable users to achieve high-accuracy semantic segmentation for diverse offroad environments, with a focus on domain generalization and robust performance.

**Key Objectives:**
- Train a state-of-the-art segmentation model on custom offroad datasets
- Implement domain generalization techniques to handle varying terrain conditions
- Achieve accurate predictions with evaluation metrics and visual comparisons
- Provide reproducible, organized workflows for research and deployment

**Target Audience:** Machine learning engineers, researchers, and developers working on computer vision tasks, particularly those involving semantic segmentation in challenging outdoor environments.

---

## b. Step-by-Step Instructions

### 1. Environment Setup
1. **Clone or Download the Project:**
   - Ensure you have the project files in your workspace (e.g., `d:\CEV`).
   - Verify the presence of key directories: `Offroad_Segmentation_Training_Dataset/`, `segformer/`, `ENV_SETUP/`.

2. **Install Dependencies:**
   - Run the setup scripts in `ENV_SETUP/`:
     - `setup_env.bat` (for environment creation)
     - `install_packages.bat` (for package installation)
   - Alternatively, manually install from `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```
   - Required packages include: PyTorch, Transformers, OpenCV, Matplotlib, etc.

3. **Validate Setup:**
   - Run `setup_validation.py` to check GPU availability, package versions, and data integrity.
   - Ensure CUDA is available if using GPU acceleration.

### 2. Data Preparation
1. **Dataset Structure:**
   - Organize data in the following format:
     ```
     Offroad_Segmentation_Training_Dataset/
     ├── train/
     │   ├── Color_Images/  # RGB images
     │   └── Segmentation/  # Ground truth masks
     └── val/
         ├── Color_Images/
         └── Segmentation/
     ```

2. **Class Mapping:**
   - The model uses 9 classes with remapped IDs:
     - 0: Trees, 1: Lush Bushes, 2: Dry Grass, 3: Dry Bushes, 4: Ground Clutter, 5: Flowers, 6: Logs, 7: Rocks, 8: Landscape, 9: Sky
   - Masks are automatically remapped during loading using predefined mappings.

3. **Preprocessing:**
   - Images are resized to 512x512, normalized with ImageNet mean/std.
   - Masks are converted to class IDs and resized with nearest-neighbor interpolation.

### 3. Model Training
1. **Phase 1 Training (Head Fine-Tuning):**
   - Run `train_segformer.py` with default settings.
   - Freezes the backbone, trains only the decoder head for 10 epochs.
   - Command: `python train_segformer.py --phase 1`

2. **Phase 2 Training (Full Fine-Tuning):**
   - Unfreezes the backbone, trains the entire model for additional epochs.
   - Uses lower learning rate for backbone (6e-5) and higher for head (6e-4).
   - Command: `python train_segformer.py --phase 2 --resume_from best_model_phase1.pth`

3. **Training Parameters:**
   - Optimizer: AdamW with weight decay 0.01
   - Loss: Weighted Cross-Entropy (0.7) + Dice Loss (0.3)
   - Scheduler: Polynomial decay
   - Batch size: 4-8 (adjust based on GPU memory)

### 4. Model Testing and Evaluation
1. **Run Inference:**
   - Use `test_segformer.py` to generate predictions.
   - Default uses `best_model_phase2.pth` with TTA enabled.
   - Command: `python test_segformer.py`

2. **Evaluation Metrics:**
   - Computes Mean IoU, Pixel Accuracy, and per-class IoU.
   - Results saved to `predictions_tta/evaluation_metrics.txt`.

3. **Generate Visual Comparisons:**
   - Saves prediction masks, colorized masks, and side-by-side comparisons.
   - Outputs in `predictions_tta/masks/`, `masks_color/`, `comparisons/`.

### 5. Visualization and Analysis
1. **Run Visualization Script:**
   - Execute `visualize.py` to generate additional plots and analyses.
   - Includes class distribution, confusion matrices, etc.

2. **TensorBoard (if available):**
   - Launch TensorBoard to view training logs: `tensorboard --logdir runs/`
   - Monitor loss curves, validation metrics in real-time.

---

## c. Diagrams & Visuals

### Pipeline Flowchart
```
[Data Loading] → [Preprocessing] → [Model Training] → [Evaluation] → [Visualization]
     ↓              ↓              ↓              ↓              ↓
[Augmentation]  [Normalization]  [Two-Phase]  [TTA Inference]  [Comparisons]
```

### Data Augmentation Table
| Augmentation Type | Parameters | Purpose |
|-------------------|------------|---------|
| ColorJitter | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 | Domain generalization |
| RandomFlips | Horizontal p=0.5, Vertical p=0.2 | Spatial invariance |
| RandomAffine | Rotation ±10°, Scale 0.9-1.1 | Viewpoint robustness |
| GaussianBlur | σ=0.1-2.0 | Noise reduction |
| RandomGrayscale | p=0.1 | Color independence |

### Model Architecture Diagram
```
Input Image (512x512)
    ↓
SegFormer-B2 Encoder (ViT-based)
    ↓
Decoder Head (Convolutional)
    ↓
Output: 9-Class Segmentation Mask
```

### Screenshots
- **Training Setup:** Screenshot of terminal output during `train_segformer.py` execution.
- **Model Loading:** Screenshot showing successful model checkpoint loading in `test_segformer.py`.
- **Evaluation Results:** Screenshot of `evaluation_metrics.txt` content.

*(Note: Insert actual screenshots from runs/ folder here after training. If runs/ folder is not present, use outputs from `predictions_tta/` or training logs.)*

---

## d. Graphs & Charts

### Training Loss and Accuracy Trends
- **Loss Curve:** Plot showing training and validation loss over epochs.
- **IoU Trend:** Line chart of Mean IoU improvement during training.
- **Per-Class Accuracy:** Bar chart comparing accuracy across the 9 classes.

### Performance Comparisons
- **Phase 1 vs Phase 2:** Comparison of metrics before and after full fine-tuning.
- **With/Without TTA:** Bar chart showing IoU improvement with test-time augmentation.

### i. Screenshots from Training Runs
- Insert images from the `runs/` folder (e.g., TensorBoard screenshots of loss curves, validation metrics).
- Example: Screenshot of TensorBoard dashboard showing epoch-wise progress.

*(If `runs/` folder does not exist, generate and include plots from training logs or use matplotlib outputs from `visualize.py`.)*

### ii. Before & After Images
- **Correct Classifications:** Side-by-side images showing input image, ground truth mask, and accurate prediction.
- **Misclassified Objects:** Examples highlighting areas where the model failed (e.g., confusing dry grass with ground clutter).
- Use images from `predictions_tta/comparisons/` folder, such as:
  - sample_000_0000060.png.png (showing input, GT, prediction)

Example Before/After:
- **Before (Ground Truth):** Colorized mask with correct class labels.
- **After (Prediction):** Model's output with any discrepancies highlighted.

---

## Additional Notes
- **Troubleshooting:** Refer to `README.md` and `QUICKSTART.md` for common issues.
- **Customization:** Modify hyperparameters in training scripts for your specific dataset.
- **Deployment:** Use the trained model in `test_segformer.py` for inference on new images.
- **Best Practices:** Always validate on a held-out test set; use TTA for production inference.

This guide ensures a structured, reproducible approach to implementing SegFormer-B2 for offroad segmentation tasks.</content>
<parameter name="filePath">d:\CEV\SEGFORMER_COMPLETE_GUIDE.md