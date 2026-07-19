from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset


def read_manifest(path: str | Path, split: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows found for split '{split}' in {path}")
    return rows


def _blob_mask(
    height: int, width: int, rng: np.random.Generator, count: tuple[int, int]
) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for _ in range(int(rng.integers(count[0], count[1] + 1))):
        radius = int(rng.integers(max(8, width // 24), max(16, width // 8)))
        x = int(rng.integers(-radius, width + radius))
        y = int(rng.integers(-radius, height + radius))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=max(2, width // 64)))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def apply_synthetic_occlusion(
    image: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, str]:
    """Occlude a CHW RGBN image in [0, 1], leaving its label untouched."""
    image = image.copy()
    _, height, width = image.shape
    kind = str(
        rng.choice(("trees", "cloud", "shadow", "cutout", "vehicles", "haze"))
    )
    mask = np.zeros((height, width), dtype=np.float32)

    if kind == "trees":
        mask = _blob_mask(height, width, rng, (4, 10))
        alpha = 0.65 * mask
        color = np.array([0.10, 0.28, 0.08, 0.45], dtype=np.float32)[:, None, None]
        image = image * (1 - alpha) + color * alpha
    elif kind == "cloud":
        mask = _blob_mask(height, width, rng, (3, 7))
        alpha = 0.82 * mask
        color = np.array([0.95, 0.95, 0.93, 0.88], dtype=np.float32)[:, None, None]
        image = image * (1 - alpha) + color * alpha
    elif kind == "shadow":
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for _ in range(int(rng.integers(1, 4))):
            x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
            polygon = [
                (x, y),
                (x + int(rng.integers(30, 100)), y + int(rng.integers(-15, 20))),
                (x + int(rng.integers(25, 90)), y + int(rng.integers(25, 100))),
                (x + int(rng.integers(-20, 20)), y + int(rng.integers(30, 95))),
            ]
            draw.polygon(polygon, fill=255)
        mask = np.asarray(canvas, dtype=np.float32) / 255.0
        image *= 1.0 - 0.72 * mask
    elif kind == "cutout":
        box_height = int(rng.integers(height // 8, height // 3))
        box_width = int(rng.integers(width // 8, width // 3))
        y = int(rng.integers(0, height - box_height))
        x = int(rng.integers(0, width - box_width))
        mask[y : y + box_height, x : x + box_width] = 1
        fill = image.mean(axis=(1, 2), keepdims=True)
        image = image * (1 - mask) + fill * mask
    elif kind == "vehicles":
        for _ in range(int(rng.integers(8, 24))):
            box_height = int(rng.integers(3, 8))
            box_width = int(rng.integers(6, 16))
            y = int(rng.integers(0, height - box_height))
            x = int(rng.integers(0, width - box_width))
            mask[y : y + box_height, x : x + box_width] = 1
            shade = float(rng.uniform(0.15, 0.9))
            image[:, y : y + box_height, x : x + box_width] = shade
    else:
        mask[:] = 1
        strength = float(rng.uniform(0.18, 0.42))
        haze = np.array([0.84, 0.86, 0.88, 0.78], dtype=np.float32)[:, None, None]
        image = image * (1 - strength) + haze * strength

    return np.clip(image, 0, 1), mask >= 0.35, kind


def _augment_pair(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image, mask = image[:, :, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[:, ::-1, :], mask[::-1, :]
    rotations = int(rng.integers(0, 4))
    if rotations:
        image = np.rot90(image, rotations, axes=(1, 2))
        mask = np.rot90(mask, rotations, axes=(0, 1))
    if rng.random() < 0.7:
        gain = float(rng.uniform(0.82, 1.18))
        bias = float(rng.uniform(-0.06, 0.06))
        image[:3] = np.clip(image[:3] * gain + bias, 0, 1)
    if rng.random() < 0.25:
        noise = rng.normal(0, 0.015, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0, 1)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class RoadDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        mean: list[float],
        std: list[float],
        training: bool = False,
        occlusion_probability: float = 0.0,
        deterministic_occlusion: bool = False,
        seed: int = 42,
    ) -> None:
        self.rows = read_manifest(manifest, split)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self.training = training
        self.occlusion_probability = occlusion_probability
        self.deterministic_occlusion = deterministic_occlusion
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = np.load(row["image_rgbn_path"]).astype(np.float32) / 255.0
        mask = np.asarray(Image.open(row["mask_path"]), dtype=np.float32) / 255.0
        if image.shape[0] != 4:
            raise ValueError(f"Expected CHW RGBN data, got {image.shape}")

        rng_seed = self.seed + index if self.deterministic_occlusion else None
        rng = np.random.default_rng(rng_seed)
        if self.training:
            image, mask = _augment_pair(image, mask, rng)

        occlusion_mask = np.zeros_like(mask, dtype=bool)
        occlusion_kind = "none"
        if rng.random() < self.occlusion_probability:
            image, occlusion_mask, occlusion_kind = apply_synthetic_occlusion(
                image, rng
            )

        image = np.ascontiguousarray((image - self.mean) / self.std)
        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])).float(),
            "occlusion_mask": torch.from_numpy(
                np.ascontiguousarray(occlusion_mask[None])
            ).bool(),
            "tile_id": row["tile_id"],
            "aoi": row["aoi"],
            "occlusion_kind": occlusion_kind,
        }


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
