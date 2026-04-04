# SegFormer-B2 Advanced Training Pipeline

This is a comprehensive upgrade to your offroad segmentation pipeline with state-of-the-art domain generalization and training techniques.

## Key Features

### 1. **SegFormer-B2 Model with Pretrained Weights**
- Uses `microsoft/segformer-b2-pretrained` backbone
- Loads pretrained weights from `segformer/` folder
- Pretrained on ADE20K ImageNet features for better initialization

### 2. **Class ID Remapping in Dataset**
```python
CLASS_MAPPING = {
    100: 0,      # Trees
    200: 1,      # Lush Bushes
    300: 2,      # Dry Grass
    500: 3,      # Dry Bushes
    550: 4,      # Ground Clutter
    700: 5,      # Logs
    800: 6,      # Rocks
    7100: 7,     # Landscape
    10000: 8,    # Sky
}
```
Remapping is applied **at load time** in the Dataset class, so all downstream code works with consistent class IDs.

### 3. **Advanced Data Augmentation for Domain Generalization**
The #1 differentiator for handling diverse desert environments:

- **ColorJitter** (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
  - Forces model to learn **shape-based features** instead of color/texture shortcuts
  - Prevents overfitting to desert color palette
  
- **RandomHorizontalFlip** (p=0.5) & **RandomVerticalFlip** (p=0.2)
  - Spatial invariance
  
- **RandomCrop** + **RandomAffine** (rotation ±10°, translation 10%, scale 0.9-1.1)
  - Robustness to different viewpoints and compositions
  
- **GaussianBlur** (σ=0.1-2.0)
  - Reduces high-frequency noise dependency
  
- **RandomGrayscale** (p=0.1)
  - Forces learning of structure independent of color

### 4. **Weighted Loss Function for Class Imbalance**
Combines CrossEntropyLoss + Dice Loss with class weights:

```python
total_loss = 0.7 × CrossEntropy(weight=class_weights) + 0.3 × Dice
```

**Class Weights:**
```
Background: 1.0
Trees: 1.0
Lush Bushes: 1.0
Dry Grass: 1.0
Dry Bushes: 1.0
Ground Clutter (rare): 8.0×
Logs (rare): 8.0×
Rocks: 1.0
Sky (common): 0.5×
```

- Rare classes (Ground Clutter, Logs) get 8-10× weight
- Sky gets 0.5× weight (very common)
- Dice loss provides additional boundary precision

### 5. **AdamW Optimizer with Polynomial LR Decay**
Standard SegFormer training config from NVLabs:

```python
Optimizer: AdamW
  - weight_decay: 0.01
  
Layer-wise Learning Rates:
  - Backbone (encoder): 1× lr
  - Head (decoder): 10× lr
  
Scheduler: Polynomial decay
  - Power: 1.0 (linear decay)
```

### 6. **Two-Phase Training Strategy**
Prevents catastrophic forgetting of ImageNet features:

**Phase 1: Freeze Backbone, Train Head Only (10 epochs)**
- Freeze all encoder parameters
- Train decoder head with learning rate = 6e-5
- Fast convergence to task-specific features
- Warm-up phase

**Phase 2: Unfreeze All, Fine-tune End-to-End (40-80 epochs)**
- Unfreeze encoder
- Use layer-wise learning rates (backbone 1×, head 10×)
- Lower learning rate = 6e-5
- Adapt ImageNet features to your domain

Benefits:
- Preserves pre-trained ImageNet knowledge
- Faster convergence
- Better generalization to desert environments

### 7. **CutMix for Rare Classes**
Synthetically increases training examples of rare classes (Ground Clutter, Logs):

```python
CutMix Strategy:
  - Probability: 30%
  - Patch size: 64×64
  - Copies patches of rare classes to random locations
  - Increases synthetic diversity without additional data
```

Effect: Model learns to recognize rare classes in different contexts.

### 8. **Test-Time Augmentation (TTA)**
Applies 4 augmentations at inference, averages outputs:

```
1. Original image
2. Horizontal flip (inference) → flip prediction back
3. Vertical flip (inference) → flip prediction back
4. Both flips (180° rotation) → flip prediction back

Aggregation: Average softmax probabilities → take argmax
```

**Benefits:**
- +2-4% mIoU improvement with zero model changes
- Zero computational overhead during training
- Improves prediction stability and confidence

## Scripts

### Training: `train_segformer.py`

```bash
python train_segformer.py
```

**Features:**
- Loads SegFormer-B2 from `segformer/` folder
- Two-phase training (freeze backbone → finetune all)
- Saves best model after each phase
- Generates training curves and metrics

**Output:**
```
train_stats_segformer/
├── best_model_phase1.pth      # Best during phase 1
├── best_model_phase2.pth      # Best overall model
├── segformer_b2_final.pth     # Final model after all epochs
├── training_history.json      # Training metrics
└── training_curves.png        # Loss/IoU/Accuracy curves
```

### Inference: `test_segformer.py`

```bash
# With TTA (default, recommended)
python test_segformer.py

# Without TTA
python test_segformer.py --use_tta

# Custom paths
python test_segformer.py \
  --model_path path/to/model.pth \
  --data_dir path/to/test/data \
  --output_dir path/to/output
```

**Features:**
- Test-Time Augmentation (4 flips, average softmax)
- Saves raw predictions and colored visualizations
- Generates comparison images
- Per-class metrics

**Output:**
```
predictions_tta/
├── masks/              # Raw prediction masks (class IDs 0-8)
├── masks_color/        # Colored predictions (RGB visualization)
├── comparisons/        # Side-by-side comparisons (10 samples)
├── evaluation_metrics.txt
├── per_class_metrics.png
└── inference_results.json
```

## Configuration

Edit these constants in the scripts:

### Training (`train_segformer.py`)
```python
BATCH_SIZE = 32              # Adjust for GPU memory
IMG_SIZE = 512              # Input image size
PHASE1_EPOCHS = 10          # Freeze backbone epochs
PHASE2_EPOCHS = 40          # Fine-tune all epochs
LR_PHASE1 = 6e-5            # Phase 1 learning rate
LR_PHASE2 = 6e-5            # Phase 2 learning rate
WEIGHT_DECAY = 0.01         # L2 regularization
CE_WEIGHT = 0.7             # CrossEntropy weight in loss
DICE_WEIGHT = 0.3           # Dice weight in loss
```

### Data Augmentation
```python
# In get_train_transforms()
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)  # Critical!
RandomGrayscale(p=0.1)
GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
# ... etc
```

### Class Weights
```python
CLASS_WEIGHTS = torch.tensor([
    1.0,    # Background
    1.0,    # Trees
    1.0,    # Lush Bushes
    1.0,    # Dry Grass
    1.0,    # Dry Bushes
    8.0,    # Ground Clutter (rare) ← increase if still underfitting
    8.0,    # Logs (rare) ← increase if still underfitting
    1.0,    # Rocks
    0.5,    # Sky (very common) ← decrease if overfitting
])
```

## Performance Expectations

### Without TTA
- Initial: ~30-40% mIoU (random initialization)
- Phase 1 (10 epochs): ~50-60% mIoU
- Phase 2 (40 epochs): ~65-75% mIoU

### With TTA (+2-4%)
- Final: ~68-79% mIoU

*Actual results depend on dataset size and quality*

## Implementation Highlights

### Why ColorJitter is Critical
Most desert environments share:
- **Same color palette** (browns, tans, greens)
- **Similar textures** (sand, dry vegetation)

Without ColorJitter, model learns:
```
Tree = green pixels + specific texture
Log = brown pixels + specific texture
→ Fails on different desert environments!
```

With ColorJitter, model learns:
```
Tree = specific shape + structure (color-invariant)
Log = elongated structure + boundaries (texture-invariant)
→ Works across desert environments!
```

### Why Two-Phase Training
ImageNet features are powerful for natural scene understanding:
- Phase 1: Adapt to your task quickly (small learning rate on head)
- Phase 2: Fine-tune ImageNet features to your domain (layer-wise LR)

Without this, direct fine-tuning often causes:
- Catastrophic forgetting of ImageNet knowledge
- Slower convergence
- Worse generalization

### Why TTA Works
1. **Spatial robustness**: Different crops/flips see different context
2. **Epistemic uncertainty reduction**: Averaging multiple forward passes
3. **No overhead**: Only during inference, not training

Average of 4 predictions is more robust than single prediction.

## Requirements

```
torch>=2.0
torchvision>=0.15
transformers>=4.30
numpy
PIL
opencv-python
matplotlib
tqdm
```

Install with:
```bash
pip install torch torchvision transformers opencv-python tqdm
```

## Dataset Structure

Expected directory layout:
```
Offroad_Segmentation_Training_Dataset/
├── train/
│   ├── Color_Images/
│   │   ├── image_001.png
│   │   └── ...
│   └── Segmentation/
│       ├── image_001.png  (mask with values 0, 100, 200, 300, 500, 550, 700, 800, 7100, 10000)
│       └── ...
└── val/
    ├── Color_Images/
    └── Segmentation/

Offroad_Segmentation_testImages/
├── Color_Images/
└── Segmentation/

segformer/
├── config.json
├── preprocessor_config.json
├── pytorch_model.bin  (pretrained weights)
└── README.md
```

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in training script
- Reduce `IMG_SIZE` (but 512 is recommended for detail)

### Poor Performance on Rare Classes
- Increase class weights for Ground Clutter/Logs (try 10-12×)
- Increase CutMix probability (up to 0.5)
- Check if mask remapping is working correctly

### Overfitting (high train IoU, low val IoU)
- Increase weight for Sky class (0.5 → 0.8)
- Increase data augmentation (stronger ColorJitter)
- Increase Phase 1 epochs (helps with head-only training)

### Underfitting (low train IoU)
- Increase Phase 2 epochs
- Reduce weight decay (0.01 → 0.001)
- Try higher learning rate (but be careful)

## References

1. **SegFormer**: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.03722)
2. **Data Augmentation**: Best practices from CVPR papers on domain generalization
3. **Layer-wise LR**: Standard practice from fine-tuning literature
4. **TTA**: Boosting strategy used in competition-winning segmentation models

## Notes

- **ColorJitter is the #1 differentiator** for domain generalization in desert environments
- Always use CutMix for rare classes during training
- TTA gives +2-4% mIoU for free at inference time
- Two-phase training ensures better convergence and generalization
- Class weights should reflect your dataset's class distribution
