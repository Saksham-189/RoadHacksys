"""Exact metric implementation recovered from the original E001-E012 pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage


@dataclass
class MetricTotals:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    relaxed_tp: float = 0.0
    relaxed_fp: float = 0.0
    relaxed_fn: float = 0.0
    occ_tp: float = 0.0
    occ_fn: float = 0.0
    connectivity_sum: float = 0.0
    components_sum: float = 0.0
    sample_count: int = 0


def update_metrics(
    totals: MetricTotals,
    logits: torch.Tensor,
    targets: torch.Tensor,
    occlusion_masks: torch.Tensor | None = None,
    threshold: float = 0.5,
    relaxed_buffer: int = 2,
) -> MetricTotals:
    predictions = (
        torch.sigmoid(logits).detach().cpu().numpy() > threshold
    ).astype(np.uint8)
    target_array = (targets.detach().cpu().numpy() > 0.5).astype(np.uint8)
    occlusion_array = None
    if occlusion_masks is not None:
        occlusion_array = (
            occlusion_masks.detach().cpu().numpy() > 0.5
        ).astype(np.uint8)

    for index in range(predictions.shape[0]):
        prediction = predictions[index, 0]
        target = target_array[index, 0]
        totals.tp += float(np.logical_and(prediction, target).sum())
        totals.fp += float(
            np.logical_and(prediction, np.logical_not(target)).sum()
        )
        totals.fn += float(
            np.logical_and(np.logical_not(prediction), target).sum()
        )

        dilated_prediction = ndimage.binary_dilation(
            prediction, iterations=relaxed_buffer
        )
        dilated_target = ndimage.binary_dilation(target, iterations=relaxed_buffer)
        totals.relaxed_tp += float(
            np.logical_and(dilated_prediction, target).sum()
        )
        totals.relaxed_fp += float(
            np.logical_and(prediction, np.logical_not(dilated_target)).sum()
        )
        totals.relaxed_fn += float(
            np.logical_and(np.logical_not(dilated_prediction), target).sum()
        )

        labels, component_count = ndimage.label(prediction)
        predicted_pixels = float(prediction.sum())
        if component_count > 0 and predicted_pixels > 0:
            sizes = ndimage.sum(
                prediction, labels, index=np.arange(1, component_count + 1)
            )
            connectivity = float(np.max(sizes) / predicted_pixels)
        else:
            connectivity = 0.0
        totals.connectivity_sum += connectivity
        totals.components_sum += float(component_count)
        totals.sample_count += 1

        if occlusion_array is not None:
            occluded_road = np.logical_and(occlusion_array[index, 0], target)
            totals.occ_tp += float(
                np.logical_and(prediction, occluded_road).sum()
            )
            totals.occ_fn += float(
                np.logical_and(np.logical_not(prediction), occluded_road).sum()
            )
    return totals


def compute_scores(totals: MetricTotals) -> dict[str, float]:
    epsilon = 1e-6
    iou = totals.tp / (totals.tp + totals.fp + totals.fn + epsilon)
    dice = (2 * totals.tp) / (
        2 * totals.tp + totals.fp + totals.fn + epsilon
    )
    precision = totals.tp / (totals.tp + totals.fp + epsilon)
    recall = totals.tp / (totals.tp + totals.fn + epsilon)
    relaxed_iou = totals.relaxed_tp / (
        totals.relaxed_tp
        + totals.relaxed_fp
        + totals.relaxed_fn
        + epsilon
    )
    occlusion_recall = totals.occ_tp / (
        totals.occ_tp + totals.occ_fn + epsilon
    )
    connectivity_ratio = totals.connectivity_sum / max(totals.sample_count, 1)
    components_count = totals.components_sum / max(totals.sample_count, 1)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "occlusion_recall": occlusion_recall,
        "relaxed_iou": relaxed_iou,
        "connectivity_ratio": connectivity_ratio,
        "components_count": components_count,
    }


def final_score(clean: dict[str, float], occluded: dict[str, float]) -> float:
    return (
        0.20 * clean["iou"]
        + 0.20 * clean["dice"]
        + 0.15 * clean["recall"]
        + 0.20 * occluded["occlusion_recall"]
        + 0.15 * clean["connectivity_ratio"]
        + 0.10 * clean["relaxed_iou"]
    )
