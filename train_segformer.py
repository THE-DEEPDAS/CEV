"""
SegFormer-B2 training script for the Offroad segmentation challenge.
"""

import json
import math
import os
import warnings
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

CLASS_MAPPING = {
    100: 0,
    200: 1,
    300: 2,
    500: 3,
    550: 4,
    600: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9,
}

CLASS_NAMES = [
    "Trees",
    "Lush Bushes",
    "Dry Grass",
    "Dry Bushes",
    "Ground Clutter",
    "Flowers",
    "Logs",
    "Rocks",
    "Landscape",
    "Sky",
]

NUM_CLASSES = len(CLASS_NAMES)
COMPACT_CLASS_MAPPING = {class_id: class_id for class_id in range(NUM_CLASSES)}
OBSERVED_MASK_ALIASES = {
    1: 0,
    2: 1,
    3: 2,
    28: 8,
    39: 9,
}

CLASS_WEIGHTS = torch.tensor(
    [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 1.0, 1.0, 0.5],
    dtype=torch.float32,
)

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
IMG_SIZE = 512
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20
LR_PHASE1 = 1e-3
LR_PHASE2 = 6e-5
WEIGHT_DECAY = 0.01
CE_WEIGHT = 0.7
DICE_WEIGHT = 0.3
POLY_POWER = 0.9
CUTMIX_PROB = 0.3
CUTMIX_PATCH_SIZE = 96
NUM_WORKERS = 4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_mapping(mask_array, mapping):
    remapped = np.full(mask_array.shape, fill_value=-1, dtype=np.int64)
    for raw_value, class_id in mapping.items():
        remapped[mask_array == raw_value] = class_id
    return remapped


def remap_mask(mask_array):
    unique_values = set(np.unique(mask_array).tolist())
    mapping_options = [CLASS_MAPPING, COMPACT_CLASS_MAPPING, OBSERVED_MASK_ALIASES]

    for mapping in mapping_options:
        if unique_values and unique_values.issubset(set(mapping.keys())):
            return apply_mapping(mask_array, mapping).astype(np.uint8)

    merged_mapping = {}
    merged_mapping.update(CLASS_MAPPING)
    merged_mapping.update(COMPACT_CLASS_MAPPING)
    merged_mapping.update(OBSERVED_MASK_ALIASES)
    remapped = apply_mapping(mask_array, merged_mapping)

    unknown_values = sorted(set(np.unique(mask_array[remapped < 0]).tolist()))
    if unknown_values:
        raise ValueError(
            "Unknown mask values detected: "
            f"{unknown_values}. Supported ids are documented ids "
            f"{sorted(CLASS_MAPPING)}, compact ids 0..{NUM_CLASSES - 1}, "
            f"and observed grayscale aliases {sorted(OBSERVED_MASK_ALIASES)}."
        )
    return remapped.astype(np.uint8)


def ensure_min_size(image, mask, min_size):
    width, height = image.size
    scale = max(min_size / width, min_size / height, 1.0)
    if scale > 1.0:
        new_width = int(math.ceil(width * scale))
        new_height = int(math.ceil(height * scale))
        image = TF.resize(image, [new_height, new_width], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [new_height, new_width], interpolation=InterpolationMode.NEAREST)
    return image, mask


class CutMix:
    def __init__(self, rare_class_ids=None, prob=0.3, patch_size=96):
        self.rare_class_ids = rare_class_ids or [4, 5, 6]
        self.prob = prob
        self.patch_size = patch_size

    def __call__(self, image, mask):
        if np.random.rand() > self.prob:
            return image, mask

        height, width = mask.shape
        for rare_class in self.rare_class_ids:
            rare_coords = np.where(mask == rare_class)
            if len(rare_coords[0]) == 0:
                continue

            src_center_y = int(np.random.choice(rare_coords[0]))
            src_center_x = int(np.random.choice(rare_coords[1]))

            src_y0 = max(0, src_center_y - self.patch_size // 2)
            src_x0 = max(0, src_center_x - self.patch_size // 2)
            src_y1 = min(height, src_y0 + self.patch_size)
            src_x1 = min(width, src_x0 + self.patch_size)
            src_y0 = max(0, src_y1 - self.patch_size)
            src_x0 = max(0, src_x1 - self.patch_size)

            patch_height = src_y1 - src_y0
            patch_width = src_x1 - src_x0
            if patch_height <= 0 or patch_width <= 0:
                continue

            dst_y0 = np.random.randint(0, max(1, height - patch_height + 1))
            dst_x0 = np.random.randint(0, max(1, width - patch_width + 1))
            dst_y1 = dst_y0 + patch_height
            dst_x1 = dst_x0 + patch_width

            image[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1].copy()
            mask[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1].copy()

        return image, mask


class JointTrainTransform:
    def __init__(self, img_size=512):
        self.img_size = img_size
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        )
        self.blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image, mask):
        image, mask = ensure_min_size(image, mask, self.img_size)

        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        top, left, height, width = transforms.RandomCrop.get_params(image, (self.img_size, self.img_size))
        image = TF.crop(image, top, left, height, width)
        mask = TF.crop(mask, top, left, height, width)

        image = self.color_jitter(image)
        if torch.rand(1).item() < 0.1:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
        if torch.rand(1).item() < 0.3:
            image = self.blur(image)

        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


class JointEvalTransform:
    def __init__(self, img_size=512):
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image, mask):
        image = TF.resize(image, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


class OffRoadSegmentationDataset(Dataset):
    def __init__(self, data_dir, joint_transform=None, apply_cutmix=False):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir = os.path.join(data_dir, "Segmentation")
        self.joint_transform = joint_transform
        self.apply_cutmix = apply_cutmix
        self.cutmix = CutMix(prob=CUTMIX_PROB, patch_size=CUTMIX_PATCH_SIZE) if apply_cutmix else None

        self.image_files = sorted(
            [name for name in os.listdir(self.image_dir) if name.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        raw_mask = np.array(Image.open(mask_path))
        remapped_mask = remap_mask(raw_mask)

        if self.apply_cutmix:
            image_np = np.array(image)
            image_np, remapped_mask = self.cutmix(image_np, remapped_mask.copy())
            image = Image.fromarray(image_np)

        mask = Image.fromarray(remapped_mask)
        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        return image, mask.long()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (probs * target_one_hot).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return dice.mean()


class WeightedCombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.7, dice_weight=0.3, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def compute_iou_per_class(logits, targets, num_classes):
    preds = torch.argmax(logits, dim=1)
    ious = []
    for class_id in range(num_classes):
        pred_mask = preds == class_id
        target_mask = targets == class_id
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append((intersection / union).item())
    return ious


def compute_mean_iou(logits, targets, num_classes):
    return float(np.nanmean(compute_iou_per_class(logits, targets, num_classes)))


def compute_pixel_accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return ((preds == targets).sum().float() / targets.numel()).item()


def polynomial_decay_scheduler(optimizer, total_steps, power=0.9):
    total_steps = max(1, total_steps)

    def lr_lambda(current_step):
        progress = min(current_step / total_steps, 1.0)
        return (1.0 - progress) ** power

    return LambdaLR(optimizer, lr_lambda)


def create_optimizer_with_layer_wise_lr(model, base_lr, weight_decay):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("segformer.encoder"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": base_lr * 10.0},
    ]
    return optim.AdamW(param_groups, weight_decay=weight_decay)


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device, phase_name="Train"):
    model.train()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=phase_name, leave=False)

    for batch_idx, (images, masks) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else nullcontext()
        with autocast_ctx:
            outputs = model(pixel_values=images)
            logits = F.interpolate(
                outputs.logits,
                size=masks.shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            loss = loss_fn(logits, masks)
            loss_for_step = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss_for_step).backward()

        if ((batch_idx + 1) % GRAD_ACCUM_STEPS == 0) or ((batch_idx + 1) == len(loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu().item())
        with torch.no_grad():
            total_iou += compute_mean_iou(logits, masks, NUM_CLASSES)
            total_acc += compute_pixel_accuracy(logits, masks)
        num_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}", miou=f"{total_iou / num_batches:.4f}")

    return {
        "loss": total_loss / max(1, num_batches),
        "iou": total_iou / max(1, num_batches),
        "acc": total_acc / max(1, num_batches),
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0

    progress = tqdm(loader, desc="Validation", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(pixel_values=images)
        logits = F.interpolate(
            outputs.logits,
            size=masks.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        loss = loss_fn(logits, masks)

        total_loss += loss.item()
        total_iou += compute_mean_iou(logits, masks, NUM_CLASSES)
        total_acc += compute_pixel_accuracy(logits, masks)
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(1, num_batches),
        "iou": total_iou / max(1, num_batches),
        "acc": total_acc / max(1, num_batches),
    }


def plot_training_curves(history, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["train_iou"], label="Train")
    axes[0, 1].plot(history["val_iou"], label="Val")
    axes[0, 1].set_title("Mean IoU")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["train_acc"], label="Train")
    axes[1, 0].plot(history["val_acc"], label="Val")
    axes[1, 0].set_title("Pixel Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    phase_boundary = PHASE1_EPOCHS - 0.5
    for axis in axes.flat:
        axis.axvline(phase_boundary, color="red", linestyle="--", alpha=0.4)
        axis.set_xlabel("Epoch")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        1.0,
        "\n".join(
            [
                "SegFormer-B2",
                f"Batch size: {BATCH_SIZE}",
                f"Phase 1 LR: {LR_PHASE1}",
                f"Phase 2 LR: {LR_PHASE2} / {LR_PHASE2 * 10:.1e}",
                f"Loss: {CE_WEIGHT} x CE + {DICE_WEIGHT} x Dice",
                "Aug: HFlip, RandomCrop(512), ColorJitter, Grayscale, Blur, CutMix",
            ]
        ),
        va="top",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "train_stats_segformer")
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(script_dir, "Offroad_Segmentation_Training_Dataset", "train")
    val_dir = os.path.join(script_dir, "Offroad_Segmentation_Training_Dataset", "val")
    model_dir = os.path.join(script_dir, "segformer")

    print("=" * 80)
    print("SEGFORMER-B2 TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")
    print(f"Model dir: {model_dir}")
    print(f"Num classes: {NUM_CLASSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Class weights: {CLASS_WEIGHTS.tolist()}")

    train_dataset = OffRoadSegmentationDataset(
        train_dir,
        joint_transform=JointTrainTransform(IMG_SIZE),
        apply_cutmix=True,
    )
    val_dataset = OffRoadSegmentationDataset(
        val_dir,
        joint_transform=JointEvalTransform(IMG_SIZE),
        apply_cutmix=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    loss_fn = WeightedCombinedLoss(
        ce_weight=CE_WEIGHT,
        dice_weight=DICE_WEIGHT,
        class_weights=CLASS_WEIGHTS.to(device),
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_iou = -1.0
    best_phase1_path = os.path.join(output_dir, "best_model_phase1.pth")
    best_phase2_path = os.path.join(output_dir, "best_model_phase2.pth")

    print("\n" + "=" * 80)
    print("PHASE 1: HEAD-ONLY TRAINING")
    print("=" * 80)
    for param in model.segformer.encoder.parameters():
        param.requires_grad = False

    phase1_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("segformer.encoder") and parameter.requires_grad
    ]
    optimizer_phase1 = optim.AdamW(phase1_params, lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)
    phase1_steps = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * PHASE1_EPOCHS
    scheduler_phase1 = polynomial_decay_scheduler(optimizer_phase1, total_steps=phase1_steps, power=POLY_POWER)

    for epoch in range(PHASE1_EPOCHS):
        print(f"\nPhase 1 - Epoch {epoch + 1}/{PHASE1_EPOCHS}")
        train_metrics = train_epoch(model, train_loader, optimizer_phase1, scheduler_phase1, loss_fn, device, "Phase1")
        val_metrics = validate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"  Train loss={train_metrics['loss']:.4f}  miou={train_metrics['iou']:.4f}  acc={train_metrics['acc']:.4f}"
        )
        print(f"  Val   loss={val_metrics['loss']:.4f}  miou={val_metrics['iou']:.4f}  acc={val_metrics['acc']:.4f}")

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(model.state_dict(), best_phase1_path)
            print(f"  -> Saved best phase 1 model ({best_val_iou:.4f})")

    print("\n" + "=" * 80)
    print("PHASE 2: END-TO-END FINE-TUNING")
    print("=" * 80)
    for param in model.segformer.encoder.parameters():
        param.requires_grad = True

    optimizer_phase2 = create_optimizer_with_layer_wise_lr(model, LR_PHASE2, WEIGHT_DECAY)
    phase2_steps = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * PHASE2_EPOCHS
    scheduler_phase2 = polynomial_decay_scheduler(optimizer_phase2, total_steps=phase2_steps, power=POLY_POWER)

    for epoch in range(PHASE2_EPOCHS):
        print(f"\nPhase 2 - Epoch {epoch + 1}/{PHASE2_EPOCHS}")
        train_metrics = train_epoch(model, train_loader, optimizer_phase2, scheduler_phase2, loss_fn, device, "Phase2")
        val_metrics = validate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"  Train loss={train_metrics['loss']:.4f}  miou={train_metrics['iou']:.4f}  acc={train_metrics['acc']:.4f}"
        )
        print(f"  Val   loss={val_metrics['loss']:.4f}  miou={val_metrics['iou']:.4f}  acc={val_metrics['acc']:.4f}")

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(model.state_dict(), best_phase2_path)
            print(f"  -> Saved best phase 2 model ({best_val_iou:.4f})")

    final_model_path = os.path.join(output_dir, "segformer_b2_final.pth")
    torch.save(model.state_dict(), final_model_path)

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    plot_training_curves(history, os.path.join(output_dir, "training_curves.png"))

    summary = {
        "model": "SegFormer-B2",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_weights": CLASS_WEIGHTS.tolist(),
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "phase1_lr": LR_PHASE1,
        "phase2_backbone_lr": LR_PHASE2,
        "phase2_head_lr": LR_PHASE2 * 10.0,
        "best_val_iou": best_val_iou,
        "mask_aliases": OBSERVED_MASK_ALIASES,
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTraining complete.")
    print(f"Best validation mIoU: {best_val_iou:.4f}")
    print(f"Final model: {final_model_path}")
    print(f"Best phase 2 model: {best_phase2_path}")


if __name__ == "__main__":
    main()
