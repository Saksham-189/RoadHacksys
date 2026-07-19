from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from roadseg.config import load_config
from roadseg.data import RoadDataset, seed_worker
from roadseg.losses import BCEDiceLoss
from roadseg.metrics import (
    binary_counts,
    metrics_from_counts,
    select_threshold,
)
from roadseg.model import PretrainedSegFormer, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_loader(
    config: dict, split: str, training: bool, occlusion_probability: float
) -> DataLoader:
    data = config["data"]
    dataset = RoadDataset(
        config["manifest"],
        split,
        data["mean"],
        data["std"],
        training=training,
        occlusion_probability=occlusion_probability,
        seed=config["seed"],
    )
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(
        dataset,
        batch_size=data["batch_size"],
        shuffle=training,
        num_workers=data["workers"],
        pin_memory=True,
        persistent_workers=data["workers"] > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    accumulation_steps: int,
    gradient_clip: float,
    amp: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = criterion(model(image), target) / accumulation_steps
        scaler.scale(loss).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.detach().item() * accumulation_steps * image.shape[0]
    return running_loss / len(loader.dataset)


@torch.inference_mode()
def validation_probabilities(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probabilities, targets = [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        probabilities.append(model(image).sigmoid().cpu())
        targets.append(batch["mask"].bool())
    return torch.cat(probabilities), torch.cat(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train E012 SegFormer RGBN")
    parser.add_argument("--config", default="configs/e012_segformer_rgbn.yaml")
    parser.add_argument("--epochs", type=int, help="Override configured epochs")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))

    train_loader = make_loader(
        config, "train", True, config["data"]["occlusion_probability"]
    )
    validation_loader = make_loader(config, "val", False, 0.0)
    model = build_model(config["model"]).to(device)
    criterion = BCEDiceLoss().to(device)
    training = config["training"]
    if isinstance(model, PretrainedSegFormer):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.encoder_parameters,
                    "lr": training["encoder_learning_rate"],
                },
                {
                    "params": model.head_parameters,
                    "lr": training["head_learning_rate"],
                },
            ],
            weight_decay=training["weight_decay"],
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training["epochs"], eta_min=1e-6
    )
    use_amp = bool(training["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} parameters={parameter_count:,} "
        f"train={len(train_loader.dataset)} val={len(validation_loader.dataset)}",
        flush=True,
    )
    history = []
    best_dice = -1.0
    stale_epochs = 0
    start = time.time()
    checkpoint_path = output / "best.pt"

    for epoch in range(1, training["epochs"] + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            training["accumulation_steps"],
            training["gradient_clip"],
            use_amp,
        )
        probabilities, targets = validation_probabilities(
            model, validation_loader, device
        )
        evaluation = config["evaluation"]
        validation_threshold, validation_dice = select_threshold(
            probabilities,
            targets,
            evaluation["threshold_min"],
            evaluation["threshold_max"],
            evaluation["threshold_steps"],
        )
        validation = metrics_from_counts(
            *binary_counts(probabilities >= validation_threshold, targets)
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_iou": validation["iou"],
            "val_dice": validation["dice"],
            "val_recall": validation["recall"],
            "val_threshold": validation_threshold,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} loss={train_loss:.5f} "
            f"val_iou={validation['iou']:.4f} "
            f"val_dice={validation['dice']:.4f} "
            f"val_recall={validation['recall']:.4f} "
            f"threshold={validation_threshold:.3f}",
            flush=True,
        )
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": validation,
                    "config": config,
                    "threshold": validation_threshold,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= training["patience"]:
            print(f"early_stopping epoch={epoch}", flush=True)
            break

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    probabilities, targets = validation_probabilities(
        model, validation_loader, device
    )
    evaluation = config["evaluation"]
    threshold, calibrated_dice = select_threshold(
        probabilities,
        targets,
        evaluation["threshold_min"],
        evaluation["threshold_max"],
        evaluation["threshold_steps"],
    )
    checkpoint["threshold"] = threshold
    checkpoint["calibrated_val_dice"] = calibrated_dice
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "device": str(device),
        "parameters": parameter_count,
        "best_epoch": checkpoint["epoch"],
        "best_val_dice_calibrated": best_dice,
        "calibrated_threshold": threshold,
        "calibrated_val_dice": calibrated_dice,
        "elapsed_minutes": (time.time() - start) / 60,
        "checkpoint": str(checkpoint_path),
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
