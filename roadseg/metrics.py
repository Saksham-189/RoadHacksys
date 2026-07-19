from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F


def binary_counts(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[int, int, int, int]:
    prediction = prediction.bool()
    target = target.bool()
    true_positive = int((prediction & target).sum())
    false_positive = int((prediction & ~target).sum())
    false_negative = int((~prediction & target).sum())
    true_negative = int((~prediction & ~target).sum())
    return true_positive, false_positive, false_negative, true_negative


def metrics_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> dict[str, float]:
    epsilon = 1e-7
    return {
        "iou": true_positive / (true_positive + false_positive + false_negative + epsilon),
        "dice": 2
        * true_positive
        / (2 * true_positive + false_positive + false_negative + epsilon),
        "precision": true_positive / (true_positive + false_positive + epsilon),
        "recall": true_positive / (true_positive + false_negative + epsilon),
        "specificity": true_negative / (true_negative + false_positive + epsilon),
    }


def relaxed_counts(
    prediction: torch.Tensor, target: torch.Tensor, radius: int
) -> tuple[int, int, int]:
    kernel = radius * 2 + 1
    prediction = prediction.float()
    target = target.float()
    dilated_prediction = F.max_pool2d(prediction, kernel, 1, radius)
    dilated_target = F.max_pool2d(target, kernel, 1, radius)
    matched_prediction = (prediction.bool() & dilated_target.bool()).sum().item()
    matched_target = (target.bool() & dilated_prediction.bool()).sum().item()
    intersection = min(matched_prediction, matched_target)
    union = prediction.sum().item() + target.sum().item() - intersection
    return int(intersection), int(union), int(target.sum().item())


def connectivity_score(prediction: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    """Measure target-road coverage while penalizing predicted fragmentation."""
    structure = np.ones((3, 3), dtype=np.uint8)
    predicted_labels, predicted_count = ndimage.label(prediction, structure)
    target_labels, target_count = ndimage.label(target, structure)
    if target_count == 0:
        return (1.0 if predicted_count == 0 else 0.0), int(predicted_count)

    weighted_score = 0.0
    target_pixels = float(target.sum())
    for component in range(1, target_count + 1):
        region = target_labels == component
        size = float(region.sum())
        overlaps = predicted_labels[region]
        labels, counts = np.unique(overlaps[overlaps > 0], return_counts=True)
        if labels.size == 0:
            continue
        coverage = float(counts.sum()) / size
        fragmentation = 1.0 / float(labels.size)
        weighted_score += (size / target_pixels) * coverage * fragmentation
    return float(weighted_score), int(predicted_count)


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    threshold: float,
    relaxed_radius: int = 2,
) -> dict[str, float]:
    model.eval()
    counts = np.zeros(4, dtype=np.int64)
    relaxed_intersection = 0
    relaxed_union = 0
    occluded_true_positive = 0
    occluded_false_negative = 0
    connectivity = []
    components = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True).bool()
        occlusion = batch["occlusion_mask"].to(device, non_blocking=True).bool()
        prediction = model(image).sigmoid() >= threshold
        counts += np.asarray(binary_counts(prediction, target))

        intersection, union, _ = relaxed_counts(
            prediction, target, relaxed_radius
        )
        relaxed_intersection += intersection
        relaxed_union += union

        occluded_target = target & occlusion
        occluded_true_positive += int((prediction & occluded_target).sum())
        occluded_false_negative += int((~prediction & occluded_target).sum())

        for predicted_item, target_item in zip(prediction, target):
            score, count = connectivity_score(
                predicted_item[0].cpu().numpy(), target_item[0].cpu().numpy()
            )
            connectivity.append(score)
            components.append(count)

    result = metrics_from_counts(*counts.tolist())
    result["relaxed_iou"] = relaxed_intersection / max(relaxed_union, 1)
    occluded_total = occluded_true_positive + occluded_false_negative
    result["occlusion_recall"] = (
        occluded_true_positive / occluded_total if occluded_total else result["recall"]
    )
    result["connectivity_ratio"] = float(np.mean(connectivity))
    result["components_count"] = float(np.mean(components))
    result["threshold"] = threshold
    return result


def select_threshold(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    minimum: float,
    maximum: float,
    steps: int,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_dice = -1.0
    for threshold in torch.linspace(minimum, maximum, steps):
        counts = binary_counts(probabilities >= threshold, targets)
        dice = metrics_from_counts(*counts)["dice"]
        if dice > best_dice:
            best_threshold = float(threshold)
            best_dice = dice
    return best_threshold, best_dice
