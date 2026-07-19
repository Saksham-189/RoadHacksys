from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadseg.config import load_config
from roadseg.data import RoadTileDataset
from roadseg.losses import BCEDiceLoss
from roadseg.metrics import compute_scores, empty_totals, update_metrics
from roadseg.models import MODEL_SPECS, build_model


def test_metrics() -> None:
    target = torch.zeros(1, 1, 16, 16)
    pred_logits = torch.full_like(target, -12.0)
    target[:, :, 4:10, 4:10] = 1
    pred_logits[:, :, 4:10, 4:10] = 12
    totals = update_metrics(empty_totals(), pred_logits, target)
    scores = compute_scores(totals)
    assert scores["iou"] > 0.99
    assert scores["dice"] > 0.99

    shifted = torch.full_like(target, -12.0)
    shifted[:, :, 4:10, 5:11] = 12
    shifted_scores = compute_scores(update_metrics(empty_totals(), shifted, target, relaxed_buffer=2))
    assert shifted_scores["relaxed_iou"] > shifted_scores["iou"]


def test_dataset() -> None:
    ds = RoadTileDataset("data/processed/bengaluru_core/t256_s128/tile_manifest.csv", split="train", max_samples=2)
    sample = ds[0]
    assert sample["image"].shape[0] == 3
    assert sample["mask"].shape[0] == 1
    assert sample["image"].shape[-2:] == sample["mask"].shape[-2:]


def test_model_forward() -> None:
    x = torch.randn(1, 3, 64, 64)
    for name in MODEL_SPECS:
        cfg = {"name": name, "base_channels": 8, "embed_dim": 32, "depth": 1, "num_queries": 4}
        model = build_model(cfg, in_channels=3)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, 64, 64), f"{name} returned {tuple(y.shape)}"


def test_one_batch_overfit() -> None:
    cfg = load_config("configs/smoke/E001_unet_smoke.yaml")
    ds = RoadTileDataset(cfg["data"]["manifest_path"], split="train", max_samples=2)
    batch = [ds[0], ds[1]]
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    model = build_model({"name": "unet", "base_channels": 8}, in_channels=3)
    criterion = BCEDiceLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert math.isfinite(losses[-1])
    assert losses[-1] <= losses[0] + 0.05


def main() -> None:
    test_metrics()
    test_dataset()
    test_model_forward()
    test_one_batch_overfit()
    print("Smoke tests passed")


if __name__ == "__main__":
    main()
