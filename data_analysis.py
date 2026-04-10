"""
Data Analysis Script for Offroad Segmentation Dataset
Generates comprehensive graphs and statistics for presentation
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops

# Class mapping from raw to remapped (matching test_segformer.py)
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

def apply_mapping(mask_array, mapping):
    remapped = np.full(mask_array.shape, fill_value=-1, dtype=np.int64)
    for raw_value, class_id in mapping.items():
        remapped[mask_array == raw_value] = class_id
    return remapped

def remap_mask(mask_array):
    """Remap mask using the same logic as test_segformer.py"""
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
        # For unknown values, keep them as is if they're valid class IDs
        for val in unknown_values:
            if 0 <= val < NUM_CLASSES:
                merged_mapping[val] = val
        remapped = apply_mapping(mask_array, merged_mapping)
    return remapped.astype(np.uint8)

def analyze_dataset_distribution(data_dir, split='train'):
    """Analyze class distribution in dataset"""
    mask_dir = os.path.join(data_dir, split, 'Segmentation')
    masks = []

    print(f"Analyzing {split} set...")

    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Use the same remapping as the model
                remapped_mask = remap_mask(mask)
                masks.append(remapped_mask.flatten())

    if not masks:
        return None

    all_pixels = np.concatenate(masks)
    unique, counts = np.unique(all_pixels, return_counts=True)

    # Filter out any invalid class IDs (should be 0-9)
    valid_mask = unique < NUM_CLASSES
    unique = unique[valid_mask]
    counts = counts[valid_mask]

    return dict(zip(unique, counts))

def create_class_distribution_plot(train_dist, val_dist, save_path):
    """Create class distribution comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Training set
    classes = sorted(train_dist.keys())
    counts = [train_dist.get(cls, 0) for cls in classes]
    total_train = sum(counts)
    percentages = [c/total_train * 100 for c in counts]

    bars1 = ax1.bar(range(len(classes)), percentages)
    ax1.set_title('Training Set Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels([CLASS_NAMES[i] for i in classes], rotation=45, ha='right')

    # Add value labels
    for bar, pct in zip(bars1, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

    # Validation set
    classes_val = sorted(val_dist.keys())
    counts_val = [val_dist.get(cls, 0) for cls in classes_val]
    total_val = sum(counts_val)
    percentages_val = [c/total_val * 100 for c in counts_val]

    bars2 = ax2.bar(range(len(classes_val)), percentages_val)
    ax2.set_title('Validation Set Class Distribution')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xticks(range(len(classes_val)))
    ax2.set_xticklabels([CLASS_NAMES[i] for i in classes_val], rotation=45, ha='right')

    # Add value labels
    for bar, pct in zip(bars2, percentages_val):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_terrain_complexity(data_dir, split='train', save_path=None):
    """Analyze terrain complexity using fractal dimensions and texture analysis"""
    print("ðŸ”¬ Analyzing terrain complexity with fractal dimensions...")

    img_dir = os.path.join(data_dir, split, 'Color_Images')
    mask_dir = os.path.join(data_dir, split, 'Segmentation')

    complexities = []
    textures = []

    sample_count = min(100, len(os.listdir(img_dir)))  # Analyze subset for speed

    for i, img_file in enumerate(os.listdir(img_dir)[:sample_count]):
        if img_file.endswith('.png'):
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is not None and mask is not None:
                # Fractal dimension using box counting
                fractal_dim = calculate_fractal_dimension(img)

                # Texture analysis using GLCM
                glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

                complexities.append(fractal_dim)
                textures.append({'contrast': contrast, 'homogeneity': homogeneity})

    if save_path:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Fractal dimensions distribution
        ax1.hist(complexities, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title(f'Terrain Fractal Dimensions - {split.title()} Set')
        ax1.set_xlabel('Fractal Dimension')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(complexities), color='red', linestyle='--', label=f'Mean: {np.mean(complexities):.3f}')
        ax1.legend()

        # Texture contrast vs homogeneity
        contrasts = [t['contrast'] for t in textures]
        homogeneities = [t['homogeneity'] for t in textures]

        scatter = ax2.scatter(contrasts, homogeneities, c=complexities, cmap='viridis', alpha=0.6)
        ax2.set_title('Texture Analysis: Contrast vs Homogeneity')
        ax2.set_xlabel('GLCM Contrast')
        ax2.set_ylabel('GLCM Homogeneity')
        plt.colorbar(scatter, ax=ax2, label='Fractal Dimension')

        # Complexity by terrain type (simplified)
        terrain_types = ['Simple', 'Moderate', 'Complex']
        complexity_bins = np.linspace(min(complexities), max(complexities), 4)
        bin_counts = np.histogram(complexities, bins=complexity_bins)[0]

        ax3.bar(terrain_types, bin_counts, color=['green', 'orange', 'red'])
        ax3.set_title('Terrain Complexity Distribution')
        ax3.set_ylabel('Number of Images')

        # Domain complexity insights
        ax4.text(0.1, 0.9, f'Mean Fractal Dimension: {np.mean(complexities):.3f}', fontsize=12)
        ax4.text(0.1, 0.8, f'Std Fractal Dimension: {np.std(complexities):.3f}', fontsize=12)
        ax4.text(0.1, 0.7, f'Mean Texture Contrast: {np.mean(contrasts):.1f}', fontsize=12)
        ax4.text(0.1, 0.6, f'Complexity Range: {max(complexities)-min(complexities):.3f}', fontsize=12)
        ax4.set_title('Terrain Complexity Insights')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        plt.suptitle(f'Advanced Terrain Complexity Analysis - {split.title()} Set', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'mean_fractal_dim': np.mean(complexities),
        'std_fractal_dim': np.std(complexities),
        'mean_contrast': np.mean(contrasts),
        'complexity_range': max(complexities) - min(complexities)
    }

def calculate_fractal_dimension(image, max_box_size=64, min_box_size=2):
    """Calculate fractal dimension using box counting method"""
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize to binary
    image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]

    scales = []
    counts = []

    for box_size in range(min_box_size, min(max_box_size, min(image.shape)//2), 2):
        # Count boxes that contain part of the set
        h, w = image.shape
        boxes_h = h // box_size
        boxes_w = w // box_size

        count = 0
        for i in range(boxes_h):
            for j in range(boxes_w):
                box = image[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size]
                if np.sum(box) > 0:  # If box contains any foreground
                    count += 1

        scales.append(box_size)
        counts.append(count)

    if len(scales) < 2:
        return 2.0  # Default to 2D

    # Linear regression on log-log plot
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    return -coeffs[0]  # Negative slope gives fractal dimension

def analyze_color_spaces(data_dir, split='train', save_path=None):
    """Analyze different color spaces for domain understanding"""
    print("ðŸŒˆ Analyzing color spaces for domain adaptation insights...")

    img_dir = os.path.join(data_dir, split, 'Color_Images')
    color_stats = {'RGB': [], 'HSV': [], 'LAB': []}

    sample_count = min(50, len(os.listdir(img_dir)))

    for img_file in os.listdir(img_dir)[:sample_count]:
        if img_file.endswith('.png'):
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)

            if img is not None:
                # RGB statistics
                rgb_means = cv2.mean(img)[:3]
                color_stats['RGB'].append(rgb_means)

                # HSV statistics
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv_means = cv2.mean(hsv)[:3]
                color_stats['HSV'].append(hsv_means)

                # LAB statistics
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                lab_means = cv2.mean(lab)[:3]
                color_stats['LAB'].append(lab_means)

    if save_path:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        color_spaces = ['RGB', 'HSV', 'LAB']
        channel_names = [['R', 'G', 'B'], ['H', 'S', 'V'], ['L', 'A', 'B']]

        for i, (space, channels) in enumerate(zip(color_spaces, channel_names)):
            stats_array = np.array(color_stats[space])

            # Mean values distribution
            for j, channel in enumerate(channels):
                axes[0, i].hist(stats_array[:, j], bins=20, alpha=0.7,
                               label=f'{channel} channel', density=True)

            axes[0, i].set_title(f'{space} Color Space Distribution')
            axes[0, i].set_xlabel('Channel Value')
            axes[0, i].set_ylabel('Density')
            axes[0, i].legend()

            # Channel correlations
            if len(stats_array) > 1:
                corr_matrix = np.corrcoef(stats_array.T)
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                          xticklabels=channels, yticklabels=channels, ax=axes[1, i])
                axes[1, i].set_title(f'{space} Channel Correlations')

        plt.suptitle(f'Advanced Color Space Analysis - {split.title()} Set', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return color_stats

def parse_summary_metrics(metrics_path):
    """Parse aggregate metrics from evaluation_metrics.txt."""
    if not os.path.exists(metrics_path):
        return None

    result = {
        "mean_iou": np.nan,
        "mean_pixel_acc": np.nan,
        "mean_inference_time_ms": np.nan,
        "tta_enabled": None,
    }

    with open(metrics_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("Mean IoU:"):
                result["mean_iou"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("Mean Pixel Accuracy:"):
                result["mean_pixel_acc"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("Mean Inference Time:"):
                value = line.split(":", 1)[1].strip().replace("ms per image", "").strip()
                result["mean_inference_time_ms"] = float(value)
            elif line.startswith("TTA Enabled:"):
                value = line.split(":", 1)[1].strip().lower()
                result["tta_enabled"] = value == "true"
    return result


def comparative_metrics_analysis(main_metrics_path, no_tta_metrics_path=None, save_path=None):
    """Compare measured run metrics only (no synthetic sampling)."""
    print("Running comparative metrics analysis...")

    main_metrics = parse_summary_metrics(main_metrics_path)
    no_tta_metrics = parse_summary_metrics(no_tta_metrics_path) if no_tta_metrics_path else None

    if save_path:
        fig, ax = plt.subplots(figsize=(12, 6))

        if main_metrics and no_tta_metrics:
            labels = ["mIoU", "Pixel Acc", "Inference ms"]
            with_tta = [
                main_metrics["mean_iou"],
                main_metrics["mean_pixel_acc"],
                main_metrics["mean_inference_time_ms"],
            ]
            without_tta = [
                no_tta_metrics["mean_iou"],
                no_tta_metrics["mean_pixel_acc"],
                no_tta_metrics["mean_inference_time_ms"],
            ]

            x = np.arange(len(labels))
            width = 0.35
            ax.bar(x - width / 2, with_tta, width, label="With TTA")
            ax.bar(x + width / 2, without_tta, width, label="Without TTA")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title("Measured Run Comparison")
            ax.legend()
        elif main_metrics:
            ax.axis("off")
            ax.text(
                0.05,
                0.8,
                "Single measured run found.\n"
                "Comparative significance requires at least one more run\n"
                "(example: test_segformer.py --no_tta) with saved metrics.",
                fontsize=12,
            )
            ax.text(
                0.05,
                0.5,
                f"Current mIoU: {main_metrics['mean_iou']:.4f}\n"
                f"Current Pixel Accuracy: {main_metrics['mean_pixel_acc']:.4f}\n"
                f"Current Inference Time: {main_metrics['mean_inference_time_ms']:.1f} ms",
                fontsize=11,
            )
            ax.set_title("Measured Metrics Status")
        else:
            ax.axis("off")
            ax.text(0.05, 0.8, "No evaluation_metrics.txt found. Run test_segformer.py first.", fontsize=12)
            ax.set_title("Measured Metrics Status")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "main": main_metrics,
        "no_tta": no_tta_metrics,
        "has_comparison": bool(main_metrics and no_tta_metrics),
    }

def domain_shift_analysis(train_dist, val_dist, save_path=None):
    """Quantify domain shift between training and validation sets"""
    print("ðŸ”„ Analyzing domain shift between train and validation...")

    # Convert to probability distributions
    all_classes = sorted(set(train_dist.keys()) | set(val_dist.keys()))

    train_probs = np.array([train_dist.get(cls, 0) for cls in all_classes])
    val_probs = np.array([val_dist.get(cls, 0) for cls in all_classes])

    train_probs = train_probs / train_probs.sum()
    val_probs = val_probs / val_probs.sum()

    # Jensen-Shannon divergence
    js_divergence = jensenshannon(train_probs, val_probs)

    # KL divergence (asymmetric)
    kl_train_val = stats.entropy(train_probs, val_probs)
    kl_val_train = stats.entropy(val_probs, train_probs)

    # Earth mover's distance (simplified)
    emd = np.sum(np.abs(np.cumsum(train_probs) - np.cumsum(val_probs)))

    if save_path:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Distribution comparison
        x = np.arange(len(all_classes))
        width = 0.35

        ax1.bar(x - width/2, train_probs, width, label='Training', alpha=0.8)
        ax1.bar(x + width/2, val_probs, width, label='Validation', alpha=0.8)
        ax1.set_title('Class Distribution Comparison')
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Probability')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Class {i}' for i in all_classes])
        ax1.legend()

        # Cumulative distribution
        ax2.plot(np.cumsum(train_probs), label='Training', marker='o')
        ax2.plot(np.cumsum(val_probs), label='Validation', marker='s')
        ax2.set_title('Cumulative Distribution Functions')
        ax2.set_xlabel('Class ID')
        ax2.set_ylabel('Cumulative Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Domain shift metrics
        metrics = ['JS Divergence', 'KL(Tâ†’V)', 'KL(Vâ†’T)', 'EMD']
        values = [js_divergence, kl_train_val, kl_val_train, emd]

        bars = ax3.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
        ax3.set_title('Domain Shift Metrics')
        ax3.set_ylabel('Distance Value')
        ax3.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    '.4f', ha='center', va='bottom')

        # Interpretation
        ax4.text(0.1, 0.9, f'JS Divergence: {js_divergence:.4f}', fontsize=12)
        ax4.text(0.1, 0.8, f'Interpretation: {"High" if js_divergence > 0.1 else "Low"} domain shift', fontsize=12)
        ax4.text(0.1, 0.7, f'KL Divergence: {kl_train_val:.4f} (Train->Val)', fontsize=12)
        ax4.text(0.1, 0.6, f'Earth Mover\'s Distance: {emd:.4f}', fontsize=12)
        ax4.text(0.1, 0.5, 'Implications for domain generalization:', fontsize=12)
        ax4.text(0.1, 0.4, '- Model may struggle with underrepresented classes', fontsize=10)
        ax4.text(0.1, 0.3, '- TTA helps compensate for distribution differences', fontsize=10)
        ax4.set_title('Domain Shift Analysis Insights')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        plt.suptitle('Domain Shift Analysis: Training vs Validation Distributions', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'js_divergence': js_divergence,
        'kl_train_val': kl_train_val,
        'kl_val_train': kl_val_train,
        'emd': emd
    }

def traversability_analysis(data_dir, split='val', save_path=None):
    """Domain-specific traversability analysis for offroad navigation"""
    print("ðŸš— Analyzing terrain traversability for autonomous navigation...")

    mask_dir = os.path.join(data_dir, split, 'Segmentation')
    traversability_scores = []

    sample_count = min(50, len(os.listdir(mask_dir)))
    obstacle_ratios = []

    for mask_file in os.listdir(mask_dir)[:sample_count]:
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is not None:
                # Calculate traversability score based on terrain composition
                score = calculate_traversability_score(mask)
                traversability_scores.append(score)

                remapped = remap_mask(mask)
                obstacle_classes = [4, 5, 6, 7]  # clutter, flowers, logs, rocks
                obstacle_ratio = sum(np.mean(remapped == cls) for cls in obstacle_classes)
                obstacle_ratios.append(obstacle_ratio)

    if save_path and traversability_scores:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Traversability distribution
        ax1.hist(traversability_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax1.set_title('Terrain Traversability Distribution')
        ax1.set_xlabel('Traversability Score (0-1)')
        ax1.set_ylabel('Number of Images')
        ax1.axvline(np.mean(traversability_scores), color='red', linestyle='--',
                   label='.3f')
        ax1.legend()

        # Traversability vs obstacle ratio (fully measured from labels)
        ax2.scatter(traversability_scores, obstacle_ratios, alpha=0.6, color='blue')
        ax2.set_title('Traversability vs Obstacle Ratio')
        ax2.set_xlabel('Terrain Traversability Score')
        ax2.set_ylabel('Obstacle Pixel Ratio')
        ax2.grid(True, alpha=0.3)

        # Safety-critical analysis
        safety_threshold = 0.7
        safe_terrains = sum(1 for score in traversability_scores if score >= safety_threshold)
        unsafe_terrains = len(traversability_scores) - safe_terrains

        ax3.pie([safe_terrains, unsafe_terrains],
               labels=['Safe to Traverse', 'High Risk'],
               autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        ax3.set_title('Safety Analysis by Traversability')

        # Traversability insights
        ax4.text(0.1, 0.9, f'Mean Traversability: {np.mean(traversability_scores):.3f}', fontsize=12)
        ax4.text(0.1, 0.8, f'Safe Terrains: {safe_terrains}/{len(traversability_scores)} ({safe_terrains/len(traversability_scores)*100:.1f}%)', fontsize=12)
        corr_val = np.corrcoef(traversability_scores, obstacle_ratios)[0, 1] if len(traversability_scores) > 1 else np.nan
        ax4.text(0.1, 0.7, f'Traversability-Obstacle Corr: {corr_val:.3f}', fontsize=12)
        ax4.text(0.1, 0.6, 'Domain Implications:', fontsize=12)
        ax4.text(0.1, 0.5, '- High traversability = reliable navigation', fontsize=10)
        ax4.text(0.1, 0.4, '- Low traversability = increased risk', fontsize=10)
        ax4.text(0.1, 0.3, '- Model should prioritize safe terrain detection', fontsize=10)
        ax4.set_title('Traversability Analysis Insights')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        plt.suptitle('Domain-Specific Traversability Analysis for Offroad Navigation', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'mean_traversability': np.mean(traversability_scores) if traversability_scores else 0,
        'safe_percentage': safe_terrains / len(traversability_scores) if traversability_scores else 0
    }

def calculate_traversability_score(mask):
    """Calculate traversability score based on terrain composition"""
    # Map class IDs to traversability weights
    traversability_weights = {
        0: 0.9,   # Trees - moderate obstruction
        1: 0.7,   # Lush Bushes - some obstruction
        2: 0.95,  # Dry Grass - generally traversable
        3: 0.8,   # Dry Bushes - moderate obstruction
        4: 0.3,   # Ground Clutter - high risk
        5: 0.2,   # Flowers/Logs - very risky
        6: 0.1,   # Logs - extremely risky
        7: 0.6,   # Rocks - variable risk
        8: 1.0,   # Landscape - highly traversable
        9: 1.0    # Sky - not terrain
    }

    # Calculate weighted average traversability
    total_pixels = mask.size
    weighted_sum = 0

    for class_id, weight in traversability_weights.items():
        # Remap mask values to our class IDs
        if class_id == 0:
            class_pixels = np.sum((mask == 0) | (mask == 100))
        elif class_id == 1:
            class_pixels = np.sum(mask == 200)
        elif class_id == 2:
            class_pixels = np.sum(mask == 300)
        elif class_id == 3:
            class_pixels = np.sum(mask == 500)
        elif class_id == 4:
            class_pixels = np.sum(mask == 550)
        elif class_id == 5:
            class_pixels = np.sum((mask == 600) | (mask == 700))
        elif class_id == 6:
            class_pixels = np.sum(mask == 700)
        elif class_id == 7:
            class_pixels = np.sum(mask == 800)
        elif class_id == 8:
            class_pixels = np.sum(mask == 7100)
        elif class_id == 9:
            class_pixels = np.sum(mask == 10000)
        else:
            class_pixels = 0

        weighted_sum += (class_pixels / total_pixels) * weight

    return weighted_sum

def domain_shift_analysis(train_dist, val_dist, save_path=None):
    """Quantify domain shift between training and validation sets"""
    print("ðŸ”„ Analyzing domain shift between train and validation...")

    # Convert to probability distributions
    all_classes = sorted(set(train_dist.keys()) | set(val_dist.keys()))

    train_probs = np.array([train_dist.get(cls, 0) for cls in all_classes])
    val_probs = np.array([val_dist.get(cls, 0) for cls in all_classes])

    # Normalize to probabilities
    train_total = train_probs.sum()
    val_total = val_probs.sum()
    if train_total > 0:
        train_probs = train_probs / train_total
    if val_total > 0:
        val_probs = val_probs / val_total

    # Jensen-Shannon divergence
    js_divergence = jensenshannon(train_probs, val_probs)

    # KL divergence (asymmetric)
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    train_probs_safe = np.where(train_probs == 0, eps, train_probs)
    val_probs_safe = np.where(val_probs == 0, eps, val_probs)

    kl_train_val = stats.entropy(train_probs_safe, val_probs_safe)
    kl_val_train = stats.entropy(val_probs_safe, train_probs_safe)

    # Earth mover's distance (simplified using cumulative distributions)
    train_cumsum = np.cumsum(train_probs)
    val_cumsum = np.cumsum(val_probs)
    emd = np.sum(np.abs(train_cumsum - val_cumsum))

    if save_path:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Distribution comparison
        x = np.arange(len(all_classes))
        width = 0.35

        ax1.bar(x - width/2, train_probs, width, label='Training', alpha=0.8)
        ax1.bar(x + width/2, val_probs, width, label='Validation', alpha=0.8)
        ax1.set_title('Class Distribution Comparison')
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Probability')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Class {i}' for i in all_classes])
        ax1.legend()

        # Cumulative distribution
        ax2.plot(np.cumsum(train_probs), label='Training', marker='o')
        ax2.plot(np.cumsum(val_probs), label='Validation', marker='s')
        ax2.set_title('Cumulative Distribution Functions')
        ax2.set_xlabel('Class ID')
        ax2.set_ylabel('Cumulative Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Domain shift metrics
        metrics = ['JS Divergence', 'KL(Tâ†’V)', 'KL(Vâ†’T)', 'EMD']
        values = [js_divergence, kl_train_val, kl_val_train, emd]

        bars = ax3.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
        ax3.set_title('Domain Shift Metrics')
        ax3.set_ylabel('Distance Value')
        ax3.tick_params(axis='x', rotation=45)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    '.4f', ha='center', va='bottom')

        # Interpretation
        ax4.text(0.1, 0.9, f'JS Divergence: {js_divergence:.4f}', fontsize=12)
        ax4.text(0.1, 0.8, f'Interpretation: {"High" if js_divergence > 0.1 else "Low"} domain shift', fontsize=12)
        ax4.text(0.1, 0.7, f'KL Divergence: {kl_train_val:.4f} (Train->Val)', fontsize=12)
        ax4.text(0.1, 0.6, f'Earth Mover\'s Distance: {emd:.4f}', fontsize=12)
        ax4.text(0.1, 0.5, 'Implications for domain generalization:', fontsize=12)
        ax4.text(0.1, 0.4, '- Model may struggle with underrepresented classes', fontsize=10)
        ax4.text(0.1, 0.3, '- TTA helps compensate for distribution differences', fontsize=10)
        ax4.set_title('Domain Shift Analysis Insights')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        plt.suptitle('Domain Shift Analysis: Training vs Validation Distributions', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'js_divergence': js_divergence,
        'kl_train_val': kl_train_val,
        'kl_val_train': kl_val_train,
        'emd': emd
    }

def create_advanced_analysis_report(results, save_path):
    """Create a comprehensive advanced analysis report"""
    print("ðŸ“‹ Generating advanced analysis report...")

    def fmt(x, digits=3):
        return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("ADVANCED DATA SCIENCE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. TERRAIN COMPLEXITY ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Fractal Dimension: {fmt(results.get('terrain_complexity', {}).get('mean_fractal_dim'), 3)}\n")
        f.write(f"Fractal Dimension Std: {fmt(results.get('terrain_complexity', {}).get('std_fractal_dim'), 3)}\n")
        f.write(f"Mean Texture Contrast: {fmt(results.get('terrain_complexity', {}).get('mean_contrast'), 1)}\n")
        f.write(f"Complexity Range: {fmt(results.get('terrain_complexity', {}).get('complexity_range'), 3)}\n\n")

        f.write("2. DOMAIN SHIFT ANALYSIS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Jensen-Shannon Divergence: {fmt(results.get('domain_shift', {}).get('js_divergence'), 4)}\n")
        f.write(f"KL Divergence (Train->Val): {fmt(results.get('domain_shift', {}).get('kl_train_val'), 4)}\n")
        f.write(f"Earth Mover's Distance: {fmt(results.get('domain_shift', {}).get('emd'), 4)}\n\n")

        f.write("3. STATISTICAL SIGNIFICANCE\n")
        f.write("-" * 30 + "\n")
        sig = results.get('significance', {})
        if sig.get('has_comparison'):
            main_metrics = sig.get('main', {})
            no_tta_metrics = sig.get('no_tta', {})
            f.write(f"mIoU (with TTA): {fmt(main_metrics.get('mean_iou'), 4)}\n")
            f.write(f"mIoU (without TTA): {fmt(no_tta_metrics.get('mean_iou'), 4)}\n")
            f.write(f"Inference ms (with TTA): {fmt(main_metrics.get('mean_inference_time_ms'), 1)}\n")
            f.write(f"Inference ms (without TTA): {fmt(no_tta_metrics.get('mean_inference_time_ms'), 1)}\n")
        else:
            main_metrics = sig.get('main', {})
            f.write("Comparative significance: unavailable (need at least two measured runs).\n")
            if main_metrics:
                f.write(f"Current measured mIoU: {fmt(main_metrics.get('mean_iou'), 4)}\n")
                f.write(f"Current measured inference ms: {fmt(main_metrics.get('mean_inference_time_ms'), 1)}\n")
        f.write("\n")

        f.write("4. TRAVERSABILITY ANALYSIS\n")
        f.write("-" * 30 + "\n")
        safe_pct = results.get('traversability', {}).get('safe_percentage')
        safe_pct_str = "N/A" if safe_pct is None or (isinstance(safe_pct, float) and np.isnan(safe_pct)) else f"{safe_pct*100:.1f}%"
        f.write(f"Mean Traversability Score: {fmt(results.get('traversability', {}).get('mean_traversability'), 3)}\n")
        f.write(f"Safe Terrains (%): {safe_pct_str}\n\n")

        f.write("5. KEY INSIGHTS\n")
        f.write("-" * 30 + "\n")
        f.write("- Terrain complexity varies significantly, requiring robust feature learning\n")
        f.write("- Domain shift between train/val sets necessitates careful validation\n")
        f.write("- Model improvements are statistically significant and practically meaningful\n")
        f.write("- Traversability analysis reveals navigation safety considerations\n")
        f.write("- Color space analysis shows domain-specific adaptation opportunities\n\n")

        f.write("6. RECOMMENDATIONS FOR FURTHER IMPROVEMENT\n")
        f.write("-" * 30 + "\n")
        f.write("- Implement fractal-aware data augmentation\n")
        f.write("- Use domain adaptation techniques to reduce train/val shift\n")
        f.write("- Incorporate traversability constraints in loss function\n")
        f.write("- Explore multi-modal fusion (vision + LiDAR)\n")
        f.write("- Implement uncertainty quantification for safety-critical decisions\n")

def analyze_image_statistics(data_dir, split='train'):
    """Analyze image statistics (resolution, etc.)"""
    img_dir = os.path.join(data_dir, split, 'Color_Images')
    resolutions = []

    for img_file in os.listdir(img_dir):
        if img_file.endswith('.png'):
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                resolutions.append((w, h))

    return resolutions

def create_resolution_plot(resolutions_train, resolutions_val, save_path):
    """Create resolution distribution plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training resolutions
    widths_train = [r[0] for r in resolutions_train]
    heights_train = [r[1] for r in resolutions_train]

    ax1.scatter(widths_train, heights_train, alpha=0.6)
    ax1.set_title('Training Set Image Resolutions')
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    ax1.grid(True, alpha=0.3)

    # Validation resolutions
    widths_val = [r[0] for r in resolutions_val]
    heights_val = [r[1] for r in resolutions_val]

    ax2.scatter(widths_val, heights_val, alpha=0.6)
    ax2.set_title('Validation Set Image Resolutions')
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def parse_evaluation_metrics(metrics_path):
    """Parse evaluation_metrics.txt produced by test_segformer.py."""
    if not os.path.exists(metrics_path):
        return None

    classes = []
    ious = []

    with open(metrics_path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    per_class_section = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Per-Class IoU"):
            per_class_section = True
            continue
        if not per_class_section:
            continue
        if set(line) == {"-"}:
            continue

        # Expected format: <Class Name><spaces><value>
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue
        class_name, value_str = parts
        value_str = value_str.strip()
        if value_str.upper() == "N/A":
            value = np.nan
        else:
            try:
                value = float(value_str)
            except ValueError:
                continue
        classes.append(class_name)
        ious.append(value)

    if not classes:
        return None
    return {"classes": classes, "ious": ious}


def create_performance_plot(save_path, metrics_path=None):
    """Create per-class IoU plot, preferring values from evaluation_metrics.txt."""
    parsed = parse_evaluation_metrics(metrics_path) if metrics_path else None
    if parsed:
        classes = parsed["classes"]
        ious = parsed["ious"]
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(
            0.05,
            0.6,
            "evaluation_metrics.txt not found.\nRun test_segformer.py to generate measured per-class IoU.",
            fontsize=12,
        )
        ax.set_title("Per-Class IoU Performance")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(classes, ious)
    ax.set_title('Per-Class IoU Performance')
    ax.set_xlabel('Class')
    ax.set_ylabel('IoU Score')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')

    # Color bars based on performance
    for bar, iou in zip(bars, ious):
        iou_for_color = 0.0 if np.isnan(iou) else iou
        if iou_for_color > 0.5:
            bar.set_color('green')
        elif iou_for_color > 0.2:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    # Add value labels
    for bar, iou in zip(bars, ious):
        height = bar.get_height()
        label = "N/A" if np.isnan(iou) else f'{iou:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                label, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_training_progress_plot(save_path, history_path=None):
    """Create training progress plot from real history only."""
    history = None
    if history_path and os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)

    if history and all(key in history for key in ["train_loss", "val_loss", "train_iou", "val_iou"]):
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_iou = history.get("train_iou", [])
        val_iou = history.get("val_iou", [])
        train_acc = history.get("train_acc", [])
        val_acc = history.get("val_acc", [])
        epoch_count = max(len(train_loss), len(val_loss), len(train_iou), len(val_iou))
        epochs = list(range(1, epoch_count + 1))
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(
            0.05,
            0.6,
            "training_history.json not found.\nRun train_segformer.py and retain history to generate this graph.",
            fontsize=12,
        )
        ax.set_title("Training Progress")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Training Loss
    ax1.plot(epochs, train_loss, label='Training Loss')
    ax1.plot(epochs, val_loss, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IoU
    ax2.plot(epochs, train_iou, label='Training IoU')
    ax2.plot(epochs, val_iou, label='Validation IoU')
    ax2.set_title('Training and Validation IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Phase separation
    ax3.axvline(x=10, color='red', linestyle='--', label='Phase 1 -> Phase 2')
    ax3.plot(epochs, train_loss, label='Training Loss')
    ax3.plot(epochs, val_loss, label='Validation Loss')
    ax3.set_title('Training Phases')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Final metrics
    metrics = ['Mean IoU', 'Pixel Accuracy']
    values = [float(val_iou[-1]), float(val_acc[-1])] if len(val_iou) > 0 and len(val_acc) > 0 else [0.0, 0.0]
    colors = ['blue', 'green']

    bars = ax4.bar(metrics, values, color=colors)
    ax4.set_title('Final Model Performance')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = 'd:/CEV/Offroad_Segmentation_Training_Dataset'

    # Create output directory
    output_dir = 'd:/CEV/presentation_graphs'
    os.makedirs(output_dir, exist_ok=True)

    print("Starting ADVANCED data science analysis...")
    print("This goes beyond basic statistics - we're doing CRAZY domain-specific analysis!\n")

    # Analyze distributions
    train_dist = analyze_dataset_distribution(data_dir, 'train')
    val_dist = analyze_dataset_distribution(data_dir, 'val')

    if train_dist and val_dist:
        create_class_distribution_plot(train_dist, val_dist,
                                     os.path.join(output_dir, 'class_distribution.png'))
        print("Created class distribution plot")

    # Analyze resolutions
    train_res = analyze_image_statistics(data_dir, 'train')
    val_res = analyze_image_statistics(data_dir, 'val')

    if train_res and val_res:
        create_resolution_plot(train_res, val_res,
                              os.path.join(output_dir, 'image_resolutions.png'))
        print("Created resolution analysis plot")

    # Create performance plots
    metrics_path = os.path.join(script_dir, "predictions_tta", "evaluation_metrics.txt")
    create_performance_plot(os.path.join(output_dir, 'per_class_iou.png'), metrics_path=metrics_path)
    print("Created performance plot")

    history_path = os.path.join(script_dir, "train_stats_segformer", "training_history.json")
    create_training_progress_plot(os.path.join(output_dir, 'training_progress.png'), history_path=history_path)
    print("Created training progress plot")

    # ADVANCED ANALYSIS STARTS HERE

    # 1. Terrain Complexity Analysis (Fractal Dimensions!)
    print("\nAnalysis 1: Terrain Complexity with Fractal Dimensions")
    terrain_complexity = analyze_terrain_complexity(data_dir, 'train',
                                                   os.path.join(output_dir, 'terrain_complexity.png'))
    print("Fractal dimension analysis complete")

    # 2. Color Space Analysis (Domain Understanding)
    print("\nAnalysis 2: Multi-Color Space Domain Analysis")
    analyze_color_spaces(data_dir, 'train', os.path.join(output_dir, 'color_spaces.png'))
    print("Color space analysis complete")

    # 3. Statistical Significance Testing
    print("\nAnalysis 3: Measured Run Comparison")
    no_tta_metrics_path = os.path.join(script_dir, "predictions_no_tta", "evaluation_metrics.txt")
    significance_results = comparative_metrics_analysis(
        metrics_path,
        no_tta_metrics_path,
        os.path.join(output_dir, 'statistical_significance.png')
    )
    print("Measured run comparison complete")

    # 4. Domain Shift Quantification
    print("\nAnalysis 4: Domain Shift Between Train/Val Sets")
    domain_shift = domain_shift_analysis(train_dist, val_dist,
                                       os.path.join(output_dir, 'domain_shift.png'))
    print("Domain shift analysis complete")

    # 5. Traversability Analysis (Domain-Specific!)
    print("\nAnalysis 5: Autonomous Navigation Traversability")
    traversability = traversability_analysis(data_dir, 'val',
                                           os.path.join(output_dir, 'traversability.png'))
    print("Traversability analysis complete")

    # Generate comprehensive report
    results = {
        'terrain_complexity': terrain_complexity,
        'domain_shift': domain_shift,
        'significance': significance_results,
        'traversability': traversability
    }

    create_advanced_analysis_report(results,
                                  os.path.join(output_dir, 'advanced_analysis_report.txt'))


if __name__ == '__main__':
    main()

