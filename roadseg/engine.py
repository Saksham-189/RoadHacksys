from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from roadseg.config import resolve_path
from roadseg.data import RoadTileDataset
from roadseg.losses import build_loss
from roadseg.metrics import compute_scores, empty_totals, update_metrics
from roadseg.models import build_model, count_parameters, model_spec
from roadseg.scoreboard import upsert_scoreboard


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(cfg: dict[str, Any], split: str, train: bool = False, occluded_eval: bool = False) -> DataLoader:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    dataset = RoadTileDataset(
        manifest_path=data_cfg["manifest_path"],
        split=split,
        input_bands=data_cfg.get("input_bands", "RGB"),
        augment=train and data_cfg.get("augment", True),
        synthetic_occlusion=(train and data_cfg.get("synthetic_occlusion", False)) or occluded_eval,
        deterministic_occlusion=occluded_eval,
        seed=int(cfg.get("seed", 42)),
        max_samples=data_cfg.get("max_samples"),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=train,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def run_dir_for(cfg: dict[str, Any]) -> Path:
    exp_id = cfg["experiment"]["id"]
    model_name = cfg["model"]["name"]
    return resolve_path("runs") / f"{exp_id}_{model_name}"


def build_experiment_model(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    in_channels = 4 if cfg["data"].get("input_bands", "RGB").upper() == "RGBN" else 3
    model = build_model(cfg["model"], in_channels=in_channels)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * images.size(0)
        count += images.size(0)
    return total / max(count, 1)


@torch.no_grad()
def evaluate_model(model, loader, device, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    totals = empty_totals()
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        occ = batch.get("occlusion_mask")
        occ = occ.to(device) if occ is not None else None
        logits = model(images)
        update_metrics(totals, logits, masks, occ, threshold=threshold)
    return compute_scores(totals)


@torch.no_grad()
def save_prediction_overlays(model, loader, device, output_dir: Path, limit: int = 12) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].cpu().numpy()
        logits = model(images)
        preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(np.uint8)
        image_np = images.cpu().numpy()
        for i in range(images.size(0)):
            rgb = np.moveaxis(image_np[i, :3], 0, -1)
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            overlay = rgb.copy()
            pred = preds[i, 0] > 0
            truth = masks[i, 0] > 0.5
            overlay[truth] = (255, 70, 40)
            overlay[pred] = (40, 180, 255)
            both = np.logical_and(pred, truth)
            overlay[both] = (80, 255, 80)
            blended = (0.55 * rgb + 0.45 * overlay).astype(np.uint8)
            tile_id = batch["tile_id"][i]
            Image.fromarray(blended).save(output_dir / f"{tile_id}_overlay.png")
            saved += 1
            if saved >= limit:
                return


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_experiment(cfg: dict[str, Any]) -> Path:
    set_seed(int(cfg.get("seed", 42)))
    device = make_device(cfg["training"].get("device", "auto"))
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "resolved_config.json", cfg)

    model = build_experiment_model(cfg, device)
    criterion = build_loss(cfg["training"].get("loss", "BCE+Dice"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    train_loader = make_loader(cfg, cfg["data"].get("train_split", "train"), train=True)
    val_loader = make_loader(cfg, cfg["data"].get("val_split", "val"), train=False)

    best_score = -1.0
    best_epoch = 0
    patience = int(cfg["training"].get("early_stopping_patience", 8))
    epochs = int(cfg["training"].get("epochs", 40))
    stale = 0
    history_path = run_dir / "history.csv"

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_dice", "val_iou", "val_base_score"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate_model(model, val_loader, device)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_dice": f"{val_metrics['dice']:.6f}",
                    "val_iou": f"{val_metrics['iou']:.6f}",
                    "val_base_score": f"{val_metrics['base_score']:.6f}",
                }
            )
            handle.flush()

            if val_metrics["base_score"] > best_score:
                best_score = val_metrics["base_score"]
                best_epoch = epoch
                stale = 0
                checkpoint = {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "best_epoch": best_epoch,
                    "params_m": count_parameters(model),
                }
                torch.save(checkpoint, run_dir / "best.pt")
                save_json(run_dir / "best_val_metrics.json", val_metrics)
            else:
                stale += 1
                if stale >= patience:
                    break

    return run_dir / "best.pt"


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(resolve_path(path), map_location=device)
    cfg = ckpt["config"]
    model = build_experiment_model(cfg, device)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg, ckpt


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str = "test",
    occluded: bool = False,
    update_scoreboard: bool = True,
) -> dict[str, float]:
    device = make_device("auto")
    model, cfg, ckpt = load_checkpoint(checkpoint_path, device)
    loader = make_loader(cfg, split, train=False, occluded_eval=occluded)
    metrics = evaluate_model(model, loader, device)

    run_dir = Path(resolve_path(checkpoint_path)).parent
    suffix = "occluded" if occluded else "clean"
    save_json(run_dir / f"metrics_{split}_{suffix}.json", metrics)
    save_prediction_overlays(model, loader, device, run_dir / f"overlays_{split}_{suffix}")

    if update_scoreboard:
        spec = model_spec(cfg["model"]["name"])
        row = {
            "exp_id": cfg["experiment"]["id"],
            "model": spec.name,
            "family": spec.family,
            "backbone": cfg["model"].get("backbone", spec.backbone),
            "input_bands": cfg["data"].get("input_bands", "RGB"),
            "train_aois": cfg["data"].get("train_aois", ""),
            "test_aois": cfg["data"].get("test_aois", ""),
            "epochs": cfg["training"].get("epochs", ""),
            "best_epoch": ckpt.get("best_epoch", ""),
            "params_m": f"{ckpt.get('params_m', count_parameters(model)):.3f}",
            "checkpoint_path": str(Path(resolve_path(checkpoint_path)).relative_to(resolve_path("."))),
            "notes": cfg["experiment"].get("notes", ""),
        }
        if occluded:
            row["occlusion_recall"] = f"{metrics['occlusion_recall']:.6f}"
        else:
            clean_keys = [
                "iou",
                "dice",
                "precision",
                "recall",
                "relaxed_iou",
                "connectivity_ratio",
                "components_count",
                "base_score",
            ]
            row.update({key: f"{metrics[key]:.6f}" for key in clean_keys})
        upsert_scoreboard(row)
    return metrics
