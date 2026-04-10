"""
SegFormer-B2 inference script with flip and crop TTA.
Includes performance timing for response time analysis.
"""

import argparse
import os
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

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
    27: 8,   # Landscape variant
    28: 8,   # Landscape
    39: 9,   # Sky
}

COLOR_PALETTE = np.array(
    [
        [34, 139, 34],
        [0, 255, 0],
        [210, 180, 140],
        [139, 90, 43],
        [128, 128, 0],
        [255, 105, 180],
        [139, 69, 19],
        [128, 128, 128],
        [160, 82, 45],
        [135, 206, 235],
    ],
    dtype=np.uint8,
)

IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def apply_mapping(mask_array, mapping):
    remapped = np.full(mask_array.shape, fill_value=-1, dtype=np.int64)
    for raw_value, class_id in mapping.items():
        remapped[mask_array == raw_value] = class_id
    return remapped


def remap_mask(mask_array):
    unique_values = set(np.unique(mask_array).tolist())
    for mapping in (CLASS_MAPPING, COMPACT_CLASS_MAPPING, OBSERVED_MASK_ALIASES):
        if unique_values and unique_values.issubset(set(mapping.keys())):
            return apply_mapping(mask_array, mapping).astype(np.uint8)

    merged_mapping = {}
    merged_mapping.update(CLASS_MAPPING)
    merged_mapping.update(COMPACT_CLASS_MAPPING)
    merged_mapping.update(OBSERVED_MASK_ALIASES)
    remapped = apply_mapping(mask_array, merged_mapping)
    unknown_values = sorted(set(np.unique(mask_array[remapped < 0]).tolist()))
    if unknown_values:
        # Gracefully handle unknown values by mapping them to their class ID if valid
        for val in unknown_values:
            if 0 <= val < NUM_CLASSES:
                merged_mapping[val] = val
        remapped = apply_mapping(mask_array, merged_mapping)
    return remapped.astype(np.uint8)


def preprocess(image, mask, img_size=512):
    image = TF.resize(image, [img_size, img_size], interpolation=InterpolationMode.BILINEAR)
    mask = TF.resize(mask, [img_size, img_size], interpolation=InterpolationMode.NEAREST)
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    mask = torch.from_numpy(np.array(mask, dtype=np.int64))
    return image, mask


def mask_to_color(mask):
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = COLOR_PALETTE[class_id]
    return color_mask


class OffRoadSegmentationDataset(Dataset):
    def __init__(self, data_dir, img_size=512):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.mask_dir = os.path.join(data_dir, "Segmentation")
        self.img_size = img_size
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
        mask = Image.fromarray(remap_mask(np.array(Image.open(mask_path))))
        image, mask = preprocess(image, mask, self.img_size)
        return image, mask.long(), image_name


class TTASegmentor:
    def __init__(self, model, img_size=512):
        self.model = model
        self.img_size = img_size
        self.crop_size = img_size
        self.scale_size = int(img_size * 1.125)

    def _forward_logits(self, image_tensor):
        outputs = self.model(pixel_values=image_tensor)
        return F.interpolate(outputs.logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)

    @torch.no_grad()
    def __call__(self, image):
        _, _, height, width = image.shape
        logits_list = []

        logits_list.append(self._forward_logits(image))

        flipped = torch.flip(image, dims=[-1])
        logits_list.append(torch.flip(self._forward_logits(flipped), dims=[-1]))

        scaled = TF.resize(image.squeeze(0), [self.scale_size, self.scale_size], interpolation=InterpolationMode.BILINEAR)
        scaled = scaled.unsqueeze(0)

        top_left = scaled[:, :, : self.crop_size, : self.crop_size]
        tl_logits = self._forward_logits(top_left)
        tl_logits = F.interpolate(tl_logits, size=(height, width), mode="bilinear", align_corners=False)
        logits_list.append(tl_logits)

        bottom = self.scale_size - self.crop_size
        bottom_right = scaled[:, :, bottom:, bottom:]
        br_logits = self._forward_logits(bottom_right)
        br_logits = F.interpolate(br_logits, size=(height, width), mode="bilinear", align_corners=False)
        logits_list.append(br_logits)

        probs = torch.stack([F.softmax(logits, dim=1) for logits in logits_list], dim=0).mean(dim=0)
        return torch.argmax(probs, dim=1)


def compute_iou_per_class(preds, targets, num_classes):
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


def compute_mean_iou(preds, targets, num_classes):
    return float(np.nanmean(compute_iou_per_class(preds, targets, num_classes)))


def compute_pixel_accuracy(preds, targets):
    return ((preds == targets).sum().float() / targets.numel()).item()


def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, img_name):
    image = img_tensor.cpu().numpy()
    image = np.moveaxis(image, 0, -1)
    image = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    image = np.clip(image, 0, 1)

    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_color)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.suptitle(f"Sample: {img_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_summary(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "evaluation_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write("EVALUATION RESULTS\n")
        handle.write("=" * 60 + "\n")
        handle.write(f"Mean IoU:            {results['mean_iou']:.4f}\n")
        handle.write(f"Mean Pixel Accuracy: {results['mean_pixel_acc']:.4f}\n")
        handle.write(f"Mean Inference Time: {results['mean_inference_time_ms']:.1f}ms per image\n")
        handle.write(f"P50 Inference Time:  {results['p50_inference_time_ms']:.1f}ms per image\n")
        handle.write(f"P90 Inference Time:  {results['p90_inference_time_ms']:.1f}ms per image\n")
        handle.write(f"P95 Inference Time:  {results['p95_inference_time_ms']:.1f}ms per image\n")
        handle.write(f"Peak GPU Memory:     {results['peak_gpu_memory_mb']:.1f}MB\n")
        handle.write(f"TTA Enabled:         {results['tta_enabled']}\n\n")
        handle.write("Per-Class IoU\n")
        handle.write("-" * 60 + "\n")
        for class_name, class_iou in zip(CLASS_NAMES, results["class_iou"]):
            value = "N/A" if np.isnan(class_iou) else f"{class_iou:.4f}"
            handle.write(f"{class_name:<20} {value}\n")

    fig, ax = plt.subplots(figsize=(12, 6))
    valid_ious = [0.0 if np.isnan(class_iou) else class_iou for class_iou in results["class_iou"]]
    bars = ax.bar(
        range(NUM_CLASSES),
        valid_ious,
        color=[COLOR_PALETTE[class_id] / 255.0 for class_id in range(NUM_CLASSES)],
        edgecolor="black",
    )
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-Class IoU (Mean: {results['mean_iou']:.4f})")
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="SegFormer-B2 inference with TTA")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(script_dir, "train_stats_segformer", "best_model_phase2.pth"),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(script_dir, "Offroad_Segmentation_testImages"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(script_dir, "predictions_tta"),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_comparisons", type=int, default=10)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--benchmark_only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OffRoadSegmentationDataset(args.data_dir, img_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SegformerForSemanticSegmentation.from_pretrained(
        os.path.join(script_dir, "segformer"),
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    tta_segmentor = TTASegmentor(model, img_size=IMG_SIZE)

    masks_dir = os.path.join(args.output_dir, "masks")
    masks_color_dir = os.path.join(args.output_dir, "masks_color")
    comparisons_dir = os.path.join(args.output_dir, "comparisons")
    if not args.benchmark_only:
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(masks_color_dir, exist_ok=True)
        os.makedirs(comparisons_dir, exist_ok=True)

    iou_scores = []
    pixel_acc_scores = []
    all_class_ious = []
    sample_count = 0
    inference_times = []

    with torch.no_grad():
        progress = tqdm(loader, desc="Processing", unit="batch")
        for images, masks, image_names in progress:
            images = images.to(device)
            masks = masks.to(device)

            # Time the inference
            start_time = time.time()
            if not args.no_tta:
                predictions = torch.stack([tta_segmentor(images[index:index + 1]).squeeze(0) for index in range(images.shape[0])])
            else:
                logits = model(pixel_values=images).logits
                logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
                predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
            inference_time = time.time() - start_time
            inference_times.extend([inference_time / images.shape[0]] * images.shape[0])  # Per-image time

            for index in range(images.shape[0]):
                pred = predictions[index]
                target = masks[index]
                class_ious = compute_iou_per_class(pred.cpu(), target.cpu(), NUM_CLASSES)
                mean_iou = compute_mean_iou(pred.cpu(), target.cpu(), NUM_CLASSES)
                pixel_acc = compute_pixel_accuracy(pred.cpu(), target.cpu())

                iou_scores.append(mean_iou)
                pixel_acc_scores.append(pixel_acc)
                all_class_ious.append(class_ious)

                if not args.benchmark_only:
                    image_name = image_names[index]
                    base_name = os.path.splitext(image_name)[0]
                    pred_np = pred.cpu().numpy().astype(np.uint8)
                    Image.fromarray(pred_np).save(os.path.join(masks_dir, f"{base_name}_pred.png"))
                    pred_color = mask_to_color(pred_np)
                    cv2.imwrite(
                        os.path.join(masks_color_dir, f"{base_name}_pred_color.png"),
                        cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR),
                    )

                    if sample_count < args.num_comparisons:
                        save_prediction_comparison(
                            images[index].cpu(),
                            masks[index].cpu(),
                            pred.cpu(),
                            os.path.join(comparisons_dir, f"sample_{sample_count:03d}_{image_name}.png"),
                            image_name,
                        )
                        sample_count += 1

    mean_class_iou = np.nanmean(np.array(all_class_ious), axis=0)
    mean_inference_time = np.mean(inference_times) if inference_times else 0
    p50_inference_time = np.percentile(inference_times, 50) if inference_times else 0
    p90_inference_time = np.percentile(inference_times, 90) if inference_times else 0
    p95_inference_time = np.percentile(inference_times, 95) if inference_times else 0
    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )
    results = {
        "mean_iou": float(np.nanmean(iou_scores)),
        "mean_pixel_acc": float(np.mean(pixel_acc_scores)),
        "class_iou": mean_class_iou.tolist(),
        "mean_inference_time_ms": mean_inference_time * 1000,
        "p50_inference_time_ms": float(p50_inference_time * 1000),
        "p90_inference_time_ms": float(p90_inference_time * 1000),
        "p95_inference_time_ms": float(p95_inference_time * 1000),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        "tta_enabled": not args.no_tta,
    }
    save_metrics_summary(results, args.output_dir)

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    print(f"Mean Pixel Accuracy: {results['mean_pixel_acc']:.4f}")
    print(f"Mean Inference Time: {results['mean_inference_time_ms']:.1f}ms per image")
    print(f"P50 Inference Time: {results['p50_inference_time_ms']:.1f}ms per image")
    print(f"P90 Inference Time: {results['p90_inference_time_ms']:.1f}ms per image")
    print(f"P95 Inference Time: {results['p95_inference_time_ms']:.1f}ms per image")
    print(f"Peak GPU Memory: {results['peak_gpu_memory_mb']:.1f}MB")
    print(f"TTA Enabled: {results['tta_enabled']}")


if __name__ == "__main__":
    main()

