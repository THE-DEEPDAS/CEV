"""Train SegFormer-B2 for offroad semantic segmentation."""

from __future__ import annotations

import csv
import argparse
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


# =========================
# Configuration
# =========================

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = SCRIPT_DIR / "Offroad_Segmentation_Training_Dataset"
TEST_ROOT = SCRIPT_DIR / "Offroad_Segmentation_testImages"
MODEL_NAME = "nvidia/mit-b2"
LOCAL_MODEL_DIR = SCRIPT_DIR / "segformer"
RUN_DIR = SCRIPT_DIR / "runs" / "final"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
PREDICTION_DIR = RUN_DIR / "predictions"
VISUALIZATION_DIR = RUN_DIR / "visualizations"
TRAIN_LOG_PATH = RUN_DIR / "training_log.csv"
PER_CLASS_IOU_PATH = RUN_DIR / "per_class_iou.csv"
BEST_MODEL_PATH = RUN_DIR / "best_model.pt"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "latest.pt"

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
    "LushBushes",
    "DryGrass",
    "DryBushes",
    "GroundClutter",
    "Flowers",
    "Logs",
    "Rocks",
    "Landscape",
    "Sky",
]
NUM_CLASSES = len(CLASS_NAMES)
REMAP_LUT = np.zeros(10001, dtype=np.int64)
for raw_id, label_id in CLASS_MAPPING.items():
    REMAP_LUT[raw_id] = label_id

CLASS_WEIGHTS = torch.tensor(
    [1.5, 8.0, 1.0, 2.0, 6.0, 10.0, 10.0, 5.0, 1.0, 1.0],
    dtype=torch.float32,
)

IMG_SIZE = 512
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 8
NUM_WORKERS = min(8, os.cpu_count() or 4)
PIN_MEMORY = True
PREFETCH_FACTOR = 4
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20
PHASE1_LR = 6e-4
PHASE2_BACKBONE_LR = 6e-5
PHASE2_HEAD_LR = 6e-4
WEIGHT_DECAY = 0.01
FOCAL_GAMMA = 2.0
DICE_SMOOTH = 1.0
TTA_FLIP = True
# Oversample the classes that are both underrepresented and currently weak in validation.
RARE_RAW_IDS = {200, 500, 550, 600, 700, 800}
RARE_OVERSAMPLE_MULTIPLIER = 6

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

COLOR_PALETTE_BGR = np.array(
    [
        [34, 139, 34],
        [0, 255, 0],
        [140, 180, 210],
        [43, 90, 139],
        [0, 128, 128],
        [180, 105, 255],
        [19, 69, 139],
        [128, 128, 128],
        [45, 82, 160],
        [235, 206, 135],
    ],
    dtype=np.uint8,
)


# =========================
# Utility helpers
# =========================


def ensure_dirs() -> None:
    for path in [RUN_DIR, CHECKPOINT_DIR, PREDICTION_DIR, VISUALIZATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def dataloader_kwargs() -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return kwargs


def natural_key(text: str) -> List[object]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(state)


def try_compile_model(model: nn.Module) -> nn.Module:
    if hasattr(torch, "compile") and int(torch.__version__.split(".", 1)[0]) >= 2:
        try:
            return torch.compile(model)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[warn] torch.compile failed, continuing without compile: {exc}")
    return model


def load_segformer_model(num_labels: int) -> SegformerForSemanticSegmentation:
    if LOCAL_MODEL_DIR.exists():
        try:
            return SegformerForSemanticSegmentation.from_pretrained(
                str(LOCAL_MODEL_DIR),
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[warn] Failed to load local SegFormer from '{LOCAL_MODEL_DIR}': {exc}")
    try:
        return SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:  # pragma: no cover - offline fallback
        print(f"[warn] Failed to load '{MODEL_NAME}', falling back to local '{LOCAL_MODEL_DIR}': {exc}")
        return SegformerForSemanticSegmentation.from_pretrained(
            str(LOCAL_MODEL_DIR),
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )


def remap_mask(raw_mask: np.ndarray) -> np.ndarray:
    clipped = np.clip(raw_mask, 0, 10000).astype(np.int64)
    return REMAP_LUT[clipped]


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, bgr in enumerate(COLOR_PALETTE_BGR):
        colored[mask == class_id] = bgr
    return colored


def infer_original_size(image_path: Path) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    return height, width


def save_csv_row(path: Path, header: Sequence[str], row: Dict[str, object]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def latest_checkpoint_path() -> Optional[Path]:
    if LATEST_CHECKPOINT_PATH.exists():
        return LATEST_CHECKPOINT_PATH

    checkpoints = sorted(CHECKPOINT_DIR.glob("ep*.pt"), key=lambda p: int(re.search(r"ep(\d+)", p.stem).group(1)))
    return checkpoints[-1] if checkpoints else None


# =========================
# Dataset definitions
# =========================


class OffroadSegmentationDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Optional[Path] = None, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = transform
        self.image_files = sorted(
            [p.name for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
            key=natural_key,
        )
        self.mask_files = []
        if self.mask_dir is not None:
            self.mask_files = sorted(
                [p.name for p in self.mask_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
                key=natural_key,
            )
            self._validate_pairs()

    def _validate_pairs(self) -> None:
        image_set = set(self.image_files)
        mask_set = set(self.mask_files)
        if len(self.image_files) != len(self.mask_files) or image_set != mask_set:
            missing_masks = sorted(image_set - mask_set, key=natural_key)[:5]
            missing_images = sorted(mask_set - image_set, key=natural_key)[:5]
            raise AssertionError(
                "Image/mask count mismatch in dataset init. "
                f"images={len(self.image_files)}, masks={len(self.mask_files)}, "
                f"missing_masks={missing_masks}, missing_images={missing_images}"
            )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int):
        image_name = self.image_files[index]
        image_path = self.image_dir / image_name
        image = np.array(Image.open(image_path).convert("RGB"))

        if self.mask_dir is None:
            if self.transform is None:
                return image_name
            transformed = self.transform(image=image)
            return transformed["image"], image_name

        mask_path = self.mask_dir / image_name
        raw_mask = np.array(Image.open(mask_path))
        mask = remap_mask(raw_mask).astype(np.uint8)

        if self.transform is None:
            return image, mask, image_name

        transformed = self.transform(image=image, mask=mask)
        pixel_values = transformed["image"]
        masks = transformed["mask"].long()
        return pixel_values, masks, image_name


def scan_rare_class_indices(dataset: OffroadSegmentationDataset) -> List[int]:
    rare_indices: List[int] = []
    for index, image_name in enumerate(dataset.image_files):
        mask_path = dataset.mask_dir / image_name  # type: ignore[union-attr]
        raw_mask = np.array(Image.open(mask_path))
        unique_values = set(np.unique(raw_mask).tolist())
        if unique_values.intersection(RARE_RAW_IDS):
            rare_indices.append(index)
    return rare_indices


def build_train_dataset(base_dataset: OffroadSegmentationDataset, rare_indices: List[int]) -> Dataset:
    if not rare_indices:
        print(f"[warn] No rare-class images found for {sorted(RARE_RAW_IDS)}; continuing without oversampling.")
        return base_dataset

    print(f"Found {len(rare_indices)} rare-class images containing raw ids {sorted(RARE_RAW_IDS)}.")
    rare_subset = Subset(base_dataset, rare_indices)
    extra_copies = [rare_subset for _ in range(RARE_OVERSAMPLE_MULTIPLIER - 1)]
    return ConcatDataset([base_dataset, *extra_copies])


# =========================
# Albumentations transforms
# =========================


def build_train_transform():
    def grayscale_aug(p: float = 0.15):
        if hasattr(A, "RandomGrayscale"):
            return A.RandomGrayscale(p=p)
        if hasattr(A, "ToGray"):
            return A.ToGray(p=p)

        # Preserve a 3-channel image shape for older Albumentations versions.
        def _to_gray(image, **kwargs):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return np.repeat(gray[:, :, None], 3, axis=2)

        return A.Lambda(image=_to_gray, p=p)

    def crop_aug():
        if hasattr(A, "CropNonEmptyMaskIfExists"):
            return A.CropNonEmptyMaskIfExists(height=IMG_SIZE, width=IMG_SIZE, p=1.0)
        return A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE)

    return A.Compose(
        [
            A.RandomScale(scale_limit=0.5, p=0.5),
            A.PadIfNeeded(
                min_height=IMG_SIZE,
                min_width=IMG_SIZE,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            crop_aug(),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15, p=0.9),
            grayscale_aug(p=0.15),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ]
    )


def build_eval_transform():
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ToTensorV2(),
        ]
    )


# =========================
# Loss functions
# =========================


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        denominator = torch.sum(probs + one_hot, dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth))
        return dice.mean()


class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.class_weights = class_weights
        self.dice = DiceLoss(smooth=DICE_SMOOTH)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** FOCAL_GAMMA * ce).mean()
        dice = self.dice(logits, targets)
        return 0.5 * focal + 0.5 * dice


# =========================
# Metrics
# =========================


@dataclass
class MetricState:
    intersections: np.ndarray
    unions: np.ndarray
    correct: int = 0
    total: int = 0


class IoUMeter:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.state = MetricState(
            intersections=np.zeros(self.num_classes, dtype=np.float64),
            unions=np.zeros(self.num_classes, dtype=np.float64),
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach()
        targets = targets.detach()
        self.state.correct += int((preds == targets).sum().item())
        self.state.total += int(targets.numel())
        for class_id in range(self.num_classes):
            pred_mask = preds == class_id
            target_mask = targets == class_id
            intersection = torch.logical_and(pred_mask, target_mask).sum().item()
            union = torch.logical_or(pred_mask, target_mask).sum().item()
            self.state.intersections[class_id] += intersection
            self.state.unions[class_id] += union

    def compute(self) -> Tuple[List[float], float, float]:
        ious: List[float] = []
        for class_id in range(self.num_classes):
            union = self.state.unions[class_id]
            if union == 0:
                ious.append(np.nan)
            else:
                ious.append(float(self.state.intersections[class_id] / union))
        miou = float(np.nanmean(ious))
        pixel_acc = float(self.state.correct / max(1, self.state.total))
        return ious, miou, pixel_acc


# =========================
# Training / validation loops
# =========================


def forward_logits(model: nn.Module, images: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    outputs = model(pixel_values=images)
    return F.interpolate(outputs.logits, size=target_size, mode="bilinear", align_corners=False)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler],
    scheduler: Optional[PolynomialLR],
    train: bool,
) -> Tuple[float, List[float], float, float]:
    model.train(mode=train)
    meter = IoUMeter(NUM_CLASSES)
    total_loss = 0.0
    total_batches = 0
    amp_enabled = device.type == "cuda"

    if train:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(loader, leave=False, desc="train" if train else "val")
    for batch in progress:
        images, masks = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            logits = forward_logits(model, images, masks.shape[-2:])
            loss = loss_fn(logits, masks)

        if train:
            assert optimizer is not None and scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            meter.update(preds, masks)

        total_loss += float(loss.item())
        total_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    per_class_iou, miou, pixel_acc = meter.compute()
    return total_loss / max(1, total_batches), per_class_iou, miou, pixel_acc


# =========================
# Checkpointing and logging
# =========================


def save_checkpoint(model: nn.Module, epoch: int) -> Path:
    checkpoint_path = CHECKPOINT_DIR / f"ep{epoch}.pt"
    target = unwrap_model(model)
    torch.save(target.state_dict(), checkpoint_path)
    return checkpoint_path


def save_best_model(model: nn.Module, path: Path) -> None:
    target = unwrap_model(model)
    torch.save(target.state_dict(), path)


def save_training_checkpoint(
    model: nn.Module,
    epoch: int,
    phase_epoch: int,
    phase: str,
    best_val_miou: float,
    best_epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[PolynomialLR] = None,
) -> Path:
    checkpoint = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "phase_epoch": phase_epoch,
        "phase": phase,
        "best_val_miou": best_val_miou,
        "best_epoch": best_epoch,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint_path = CHECKPOINT_DIR / f"ep{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, LATEST_CHECKPOINT_PATH)
    return checkpoint_path


def load_training_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[PolynomialLR] = None,
) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint

    # Backward compatibility with older weight-only checkpoints.
    unwrap_model(model).load_state_dict(checkpoint)
    return {
        "epoch": 0,
        "phase": "phase1",
        "best_val_miou": -1.0,
        "best_epoch": -1,
    }


def append_epoch_logs(
    epoch: int,
    phase: str,
    train_loss: float,
    train_miou: float,
    val_loss: float,
    val_miou: float,
    lr: float,
    per_class_iou: List[float],
) -> None:
    save_csv_row(
        TRAIN_LOG_PATH,
        ["phase", "epoch", "train_loss", "train_miou", "val_loss", "val_miou", "lr"],
        {
            "phase": phase,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "lr": lr,
        },
    )

    per_class_row = {"epoch": epoch}
    for class_name, value in zip(CLASS_NAMES, per_class_iou):
        per_class_row[class_name] = value
    save_csv_row(PER_CLASS_IOU_PATH, ["epoch", *CLASS_NAMES], per_class_row)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(max(group["lr"] for group in optimizer.param_groups))


# =========================
# Test-time augmentation inference
# =========================


@torch.no_grad()
def tta_predict(model: nn.Module, image: torch.Tensor, original_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    amp_enabled = device.type == "cuda"

    def logits_from(x: torch.Tensor) -> torch.Tensor:
        logits = forward_logits(model, x, x.shape[-2:])
        return logits

    inputs = [image]
    if TTA_FLIP:
        inputs.append(torch.flip(image, dims=[3]))

    logits_list = []
    for inp in inputs:
        with autocast(enabled=amp_enabled):
            logits = logits_from(inp)
        if inp is not image and TTA_FLIP:
            logits = torch.flip(logits, dims=[3])
        logits_list.append(F.softmax(logits, dim=1))

    avg_probs = torch.stack(logits_list, dim=0).mean(dim=0)
    pred = torch.argmax(avg_probs, dim=1, keepdim=True).float()
    pred = F.interpolate(pred, size=original_size, mode="nearest").squeeze(0).squeeze(0).to(torch.uint8)
    return pred.cpu()


def save_prediction_artifacts(pred_mask: torch.Tensor, base_name: str) -> None:
    pred_np = pred_mask.numpy().astype(np.uint8)
    Image.fromarray(pred_np).save(PREDICTION_DIR / f"{base_name}.png")
    color_mask = mask_to_color(pred_np)
    cv2.imwrite(str(VISUALIZATION_DIR / f"{base_name}.png"), color_mask)


# =========================
# Main
# =========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer on the off-road dataset.")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from a specific checkpoint path instead of the latest saved checkpoint.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore any saved checkpoints and start a new run from the pretrained SegFormer weights.",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Try torch.compile for the model. Leave off on Windows if Triton/Inductor is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not torch.cuda.is_available():
        print("[warn] CUDA unavailable. Falling back to CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    print("=" * 80)
    print("SegFormer-B2 offroad training")
    print(f"Device: {device}")
    print(f"Train root: {TRAIN_ROOT}")
    print(f"Test root: {TEST_ROOT}")

    train_image_dir = TRAIN_ROOT / "train" / "Color_Images"
    train_mask_dir = TRAIN_ROOT / "train" / "Segmentation"
    val_image_dir = TRAIN_ROOT / "val" / "Color_Images"
    val_mask_dir = TRAIN_ROOT / "val" / "Segmentation"
    test_image_dir = TEST_ROOT / "Color_Images"

    train_base = OffroadSegmentationDataset(train_image_dir, train_mask_dir, transform=build_train_transform())
    val_dataset = OffroadSegmentationDataset(val_image_dir, val_mask_dir, transform=build_eval_transform())
    test_dataset = OffroadSegmentationDataset(test_image_dir, mask_dir=None, transform=build_eval_transform())

    rare_indices = scan_rare_class_indices(
        OffroadSegmentationDataset(train_image_dir, train_mask_dir, transform=None)
    )
    print(f"Rare-class images found: {len(rare_indices)}")
    train_dataset = build_train_dataset(train_base, rare_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        **dataloader_kwargs(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        **dataloader_kwargs(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        **dataloader_kwargs(),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = load_segformer_model(NUM_CLASSES)
    model.config.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    model.config.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}
    model = model.to(device)

    resume_state: Optional[Dict[str, object]] = None
    resume_path: Optional[Path] = None
    if not args.fresh:
        if args.resume_from:
            resume_path = Path(args.resume_from).expanduser()
        else:
            resume_path = latest_checkpoint_path()
        if resume_path is not None and resume_path.exists():
            print(f"[resume] Loading checkpoint: {resume_path}")
            resume_state = load_training_checkpoint(model, resume_path, device)
        elif resume_path is not None:
            print(f"[resume] Checkpoint not found, starting fresh: {resume_path}")

    if args.compile_model:
        model = try_compile_model(model)
    else:
        print("[info] torch.compile disabled; running in eager mode for maximum compatibility.")

    class_weights = CLASS_WEIGHTS.to(device)
    loss_fn = CombinedFocalDiceLoss(class_weights=class_weights)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_miou = float(resume_state.get("best_val_miou", -1.0)) if resume_state else -1.0
    best_epoch = int(resume_state.get("best_epoch", -1)) if resume_state else -1
    global_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    current_phase = str(resume_state.get("phase", "phase1")) if resume_state else "phase1"
    current_phase_epoch = int(resume_state.get("phase_epoch", 0)) if resume_state else 0
    if resume_state:
        print(
            f"[resume] phase={current_phase} phase_epoch={current_phase_epoch} "
            f"global_epoch={global_epoch} best_val_miou={best_val_miou:.4f}"
        )

    # =========================
    # Phase 1: head only
    # =========================
    print("=" * 80)
    print("Phase 1: head only")
    phase1_start_epoch = 1
    if resume_state and current_phase == "phase1":
        phase1_start_epoch = current_phase_epoch + 1

    if phase1_start_epoch <= PHASE1_EPOCHS:
        for param in model.segformer.parameters():
            param.requires_grad = False
        for param in model.decode_head.parameters():
            param.requires_grad = True

        phase1_optimizer = AdamW(model.decode_head.parameters(), lr=PHASE1_LR, weight_decay=WEIGHT_DECAY)
        if resume_state and current_phase == "phase1" and "optimizer_state_dict" in resume_state:
            phase1_optimizer.load_state_dict(resume_state["optimizer_state_dict"])  # type: ignore[arg-type]
        if resume_state and current_phase == "phase1" and "scaler_state_dict" in resume_state:
            scaler.load_state_dict(resume_state["scaler_state_dict"])  # type: ignore[arg-type]

        for phase_epoch in range(phase1_start_epoch, PHASE1_EPOCHS + 1):
            global_epoch += 1
            train_loss, _, train_miou, _ = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=phase1_optimizer,
                loss_fn=loss_fn,
                device=device,
                scaler=scaler,
                scheduler=None,
                train=True,
            )
            val_loss, val_class_iou, val_miou, _ = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                loss_fn=loss_fn,
                device=device,
                scaler=None,
                scheduler=None,
                train=False,
            )
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_epoch = global_epoch
                save_best_model(model, BEST_MODEL_PATH)
            save_training_checkpoint(
                model,
                epoch=global_epoch,
                phase_epoch=phase_epoch,
                phase="phase1",
                best_val_miou=best_val_miou,
                best_epoch=best_epoch,
                optimizer=phase1_optimizer,
                scaler=scaler,
            )
            append_epoch_logs(
                epoch=global_epoch,
                phase="phase1",
                train_loss=train_loss,
                train_miou=train_miou,
                val_loss=val_loss,
                val_miou=val_miou,
                lr=current_lr(phase1_optimizer),
                per_class_iou=val_class_iou,
            )
            print(
                f"phase=phase1 epoch={phase_epoch}/{PHASE1_EPOCHS} "
                f"train_loss={train_loss:.4f} train_miou={train_miou:.4f} "
                f"val_loss={val_loss:.4f} val_miou={val_miou:.4f} "
                f"best_val_miou_so_far={best_val_miou:.4f}"
            )

    # =========================
    # Phase 2: full fine-tuning
    # =========================
    print("=" * 80)
    print("Phase 2: full fine-tuning")
    for param in model.parameters():
        param.requires_grad = True

    phase2_start_epoch = 1
    if resume_state and current_phase == "phase2":
        phase2_start_epoch = current_phase_epoch + 1

    phase2_optimizer = AdamW(
        [
            {"params": model.segformer.parameters(), "lr": PHASE2_BACKBONE_LR},
            {"params": model.decode_head.parameters(), "lr": PHASE2_HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    total_iters = PHASE2_EPOCHS * max(1, len(train_loader))
    phase2_scheduler = PolynomialLR(phase2_optimizer, total_iters=total_iters, power=1.0)

    if phase2_start_epoch <= PHASE2_EPOCHS:
        if resume_state and current_phase == "phase2":
            if "optimizer_state_dict" in resume_state:
                phase2_optimizer.load_state_dict(resume_state["optimizer_state_dict"])  # type: ignore[arg-type]
            if "scheduler_state_dict" in resume_state:
                phase2_scheduler.load_state_dict(resume_state["scheduler_state_dict"])  # type: ignore[arg-type]
            if "scaler_state_dict" in resume_state:
                scaler.load_state_dict(resume_state["scaler_state_dict"])  # type: ignore[arg-type]

        for phase_epoch in range(phase2_start_epoch, PHASE2_EPOCHS + 1):
            global_epoch += 1
            train_loss, _, train_miou, _ = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=phase2_optimizer,
                loss_fn=loss_fn,
                device=device,
                scaler=scaler,
                scheduler=phase2_scheduler,
                train=True,
            )
            val_loss, val_class_iou, val_miou, _ = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                loss_fn=loss_fn,
                device=device,
                scaler=None,
                scheduler=None,
                train=False,
            )
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_epoch = global_epoch
                save_best_model(model, BEST_MODEL_PATH)
            save_training_checkpoint(
                model,
                epoch=global_epoch,
                phase_epoch=phase_epoch,
                phase="phase2",
                best_val_miou=best_val_miou,
                best_epoch=best_epoch,
                optimizer=phase2_optimizer,
                scaler=scaler,
                scheduler=phase2_scheduler,
            )
            append_epoch_logs(
                epoch=global_epoch,
                phase="phase2",
                train_loss=train_loss,
                train_miou=train_miou,
                val_loss=val_loss,
                val_miou=val_miou,
                lr=current_lr(phase2_optimizer),
                per_class_iou=val_class_iou,
            )
            print(
                f"phase=phase2 epoch={phase_epoch}/{PHASE2_EPOCHS} "
                f"train_loss={train_loss:.4f} train_miou={train_miou:.4f} "
                f"val_loss={val_loss:.4f} val_miou={val_miou:.4f} "
                f"best_val_miou_so_far={best_val_miou:.4f}"
            )

    # =========================
    # Test inference with TTA
    # =========================
    print("=" * 80)
    print("Loading best checkpoint for TTA inference")
    inference_model = load_segformer_model(NUM_CLASSES).to(device)
    load_model_checkpoint(inference_model, BEST_MODEL_PATH, device)
    inference_model.eval()

    for batch in tqdm(test_loader, desc="test"):
        images, image_names = batch
        images = images.to(device, non_blocking=True)
        for idx, image_name in enumerate(image_names):
            original_size = infer_original_size(test_image_dir / image_name)
            pred = tta_predict(inference_model, images[idx : idx + 1], original_size, device)
            base_name = Path(image_name).stem
            save_prediction_artifacts(pred, base_name)

    # =========================
    # Summary
    # =========================
    print("=" * 80)
    print(f"Best val mIoU: {best_val_miou:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
