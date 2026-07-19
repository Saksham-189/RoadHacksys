"""Original deterministic synthetic occlusion generator used by E012."""

from __future__ import annotations

import math
import random

import numpy as np


def _draw_rect(
    mask: np.ndarray,
    rng: random.Random,
    minimum: float,
    maximum: float,
) -> tuple[slice, slice]:
    height, width = mask.shape
    rect_height = max(4, int(height * rng.uniform(minimum, maximum)))
    rect_width = max(4, int(width * rng.uniform(minimum, maximum)))
    top = rng.randint(0, max(0, height - rect_height))
    left = rng.randint(0, max(0, width - rect_width))
    return slice(top, top + rect_height), slice(left, left + rect_width)


def _draw_blob(
    mask: np.ndarray,
    rng: random.Random,
    radius_range: tuple[float, float],
) -> np.ndarray:
    height, width = mask.shape
    center_y = rng.randint(0, height - 1)
    center_x = rng.randint(0, width - 1)
    radius = max(5, int(min(height, width) * rng.uniform(*radius_range)))
    rows, columns = np.ogrid[:height, :width]
    blob = (
        (rows - center_y) ** 2 + (columns - center_x) ** 2 <= radius**2
    )
    mask[blob] = 1
    return blob


def apply_legacy_occlusion(
    image: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    output = image.copy()
    _, height, width = output.shape
    occlusion = np.zeros((height, width), dtype=np.uint8)
    for _ in range(rng.randint(2, 5)):
        event = rng.choice(
            ["tree", "cloud", "shadow", "cutout", "vehicle", "haze"]
        )
        if event == "tree":
            for _ in range(rng.randint(2, 5)):
                blob = _draw_blob(occlusion, rng, (0.025, 0.08))
                green = np.array([0.08, 0.22, 0.08, 0.28], dtype=np.float32)[
                    : output.shape[0]
                ]
                output[:, blob] = output[:, blob] * 0.25 + green[:, None] * 0.75
        elif event == "cloud":
            for _ in range(rng.randint(1, 3)):
                blob = _draw_blob(occlusion, rng, (0.04, 0.12))
                cloud = np.array([0.92, 0.92, 0.88, 0.75], dtype=np.float32)[
                    : output.shape[0]
                ]
                output[:, blob] = output[:, blob] * 0.25 + cloud[:, None] * 0.75
        elif event == "shadow":
            rows, columns = _draw_rect(occlusion, rng, 0.08, 0.22)
            occlusion[rows, columns] = 1
            output[:, rows, columns] *= rng.uniform(0.15, 0.45)
        elif event == "cutout":
            rows, columns = _draw_rect(occlusion, rng, 0.04, 0.16)
            occlusion[rows, columns] = 1
            output[:, rows, columns] *= rng.uniform(0.0, 0.18)
        elif event == "vehicle":
            for _ in range(rng.randint(4, 10)):
                rows, columns = _draw_rect(occlusion, rng, 0.012, 0.035)
                occlusion[rows, columns] = 1
                output[:, rows, columns] = rng.choice([0.05, 0.85, 0.6])
        else:
            rows = np.linspace(-1, 1, height)[:, None]
            columns = np.linspace(-1, 1, width)[None, :]
            haze = np.exp(
                -(columns**2 + rows**2) / rng.uniform(0.8, 1.8)
            )
            haze = haze > rng.uniform(0.45, 0.75)
            occlusion[haze] = 1
            output[:, haze] = output[:, haze] * 0.55 + 0.38

    if rng.random() < 0.4:
        angle = rng.uniform(0, math.pi)
        rows, columns = np.ogrid[:height, :width]
        line = np.abs(
            (columns - width / 2) * math.cos(angle)
            + (rows - height / 2) * math.sin(angle)
        )
        stripe = line < rng.uniform(6, 18)
        occlusion[stripe] = 1
        output[:, stripe] *= rng.uniform(0.2, 0.5)
    return np.clip(output, 0, 1), occlusion
