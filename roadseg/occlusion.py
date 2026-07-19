from __future__ import annotations

import math
import random

import numpy as np


def _clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def _draw_rect(mask: np.ndarray, rng: random.Random, min_frac: float, max_frac: float) -> tuple[slice, slice]:
    h, w = mask.shape
    rect_h = max(4, int(h * rng.uniform(min_frac, max_frac)))
    rect_w = max(4, int(w * rng.uniform(min_frac, max_frac)))
    y0 = rng.randint(0, max(0, h - rect_h))
    x0 = rng.randint(0, max(0, w - rect_w))
    return slice(y0, y0 + rect_h), slice(x0, x0 + rect_w)


def _draw_blob(mask: np.ndarray, rng: random.Random, radius_frac: tuple[float, float]) -> np.ndarray:
    h, w = mask.shape
    cy = rng.randint(0, h - 1)
    cx = rng.randint(0, w - 1)
    radius = max(5, int(min(h, w) * rng.uniform(*radius_frac)))
    yy, xx = np.ogrid[:h, :w]
    blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    mask[blob] = 1
    return blob


def apply_synthetic_occlusion(
    image: np.ndarray,
    rng: random.Random | None = None,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply synthetic occlusions to CxHxW float image and return occlusion mask."""
    rng = rng or random.Random()
    out = image.copy()
    _, h, w = out.shape
    occ = np.zeros((h, w), dtype=np.uint8)
    n_events = rng.randint(2, 5)

    for _ in range(n_events):
        event = rng.choice(["tree", "cloud", "shadow", "cutout", "vehicle", "haze"])
        if event == "tree":
            for _ in range(rng.randint(2, 5)):
                blob = _draw_blob(occ, rng, (0.025, 0.08))
                green = np.array([0.08, 0.22, 0.08, 0.28], dtype=np.float32)[: out.shape[0]]
                out[:, blob] = out[:, blob] * 0.25 + green[:, None] * 0.75
        elif event == "cloud":
            for _ in range(rng.randint(1, 3)):
                blob = _draw_blob(occ, rng, (0.04, 0.12))
                cloud = np.array([0.92, 0.92, 0.88, 0.75], dtype=np.float32)[: out.shape[0]]
                out[:, blob] = out[:, blob] * 0.25 + cloud[:, None] * 0.75
        elif event == "shadow":
            ys, xs = _draw_rect(occ, rng, 0.08, 0.22)
            occ[ys, xs] = 1
            out[:, ys, xs] *= rng.uniform(0.15, 0.45) * strength
        elif event == "cutout":
            ys, xs = _draw_rect(occ, rng, 0.04, 0.16)
            occ[ys, xs] = 1
            out[:, ys, xs] *= rng.uniform(0.0, 0.18)
        elif event == "vehicle":
            for _ in range(rng.randint(4, 10)):
                ys, xs = _draw_rect(occ, rng, 0.012, 0.035)
                occ[ys, xs] = 1
                color = rng.choice([0.05, 0.85, 0.6])
                out[:, ys, xs] = color
        else:
            yy = np.linspace(-1, 1, h)[:, None]
            xx = np.linspace(-1, 1, w)[None, :]
            haze = np.exp(-(xx**2 + yy**2) / rng.uniform(0.8, 1.8))
            haze = haze > rng.uniform(0.45, 0.75)
            occ[haze] = 1
            out[:, haze] = out[:, haze] * 0.55 + 0.38

    if rng.random() < 0.4:
        angle = rng.uniform(0, math.pi)
        yy, xx = np.ogrid[:h, :w]
        line = np.abs((xx - w / 2) * math.cos(angle) + (yy - h / 2) * math.sin(angle))
        stripe = line < rng.uniform(6, 18)
        occ[stripe] = 1
        out[:, stripe] *= rng.uniform(0.2, 0.5)

    return _clip01(out), occ

