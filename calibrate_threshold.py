from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from roadseg.data import RoadDataset
from roadseg.metrics import (
    binary_counts,
    connectivity_score,
    metrics_from_counts,
    relaxed_counts,
)
from roadseg.model import build_model


@torch.inference_mode()
def collect(
    model: torch.nn.Module,
    dataset: RoadDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities, targets, occlusions = [], [], []
    model.eval()
    for batch in loader:
        probabilities.append(model(batch["image"].to(device)).sigmoid().cpu())
        targets.append(batch["mask"].bool())
        occlusions.append(batch["occlusion_mask"].bool())
    return (
        torch.cat(probabilities),
        torch.cat(targets),
        torch.cat(occlusions),
    )


def calculate(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    occlusions: torch.Tensor,
    threshold: float,
    radius: int,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    result = metrics_from_counts(*binary_counts(predictions, targets))
    intersection, union, _ = relaxed_counts(predictions, targets, radius)
    result["relaxed_iou"] = intersection / max(union, 1)
    occluded_targets = targets & occlusions
    occluded_total = int(occluded_targets.sum())
    result["occlusion_recall"] = (
        int((predictions & occluded_targets).sum()) / occluded_total
        if occluded_total
        else result["recall"]
    )
    connectivity = [
        connectivity_score(prediction[0].numpy(), target[0].numpy())[0]
        for prediction, target in zip(predictions, targets)
    ]
    result["connectivity_ratio"] = float(np.mean(connectivity))
    return result


def score(clean: dict[str, float], occluded: dict[str, float]) -> float:
    return (
        0.20 * clean["iou"]
        + 0.20 * clean["dice"]
        + 0.15 * clean["recall"]
        + 0.20 * occluded["occlusion_recall"]
        + 0.15 * clean["connectivity_ratio"]
        + 0.10 * clean["relaxed_iou"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="runs/e013_pretrained_segformer_rgbn/best.pt"
    )
    parser.add_argument("--minimum", type=float, default=0.10)
    parser.add_argument("--maximum", type=float, default=0.55)
    parser.add_argument("--steps", type=int, default=19)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    data = config["data"]
    common = {
        "manifest": config["manifest"],
        "split": "val",
        "mean": data["mean"],
        "std": data["std"],
        "training": False,
        "seed": config["seed"] + 20_000,
    }
    clean_dataset = RoadDataset(**common)
    occluded_dataset = RoadDataset(
        **common, occlusion_probability=1.0, deterministic_occlusion=True
    )
    clean_data = collect(model, clean_dataset, device, data["batch_size"])
    occluded_data = collect(model, occluded_dataset, device, data["batch_size"])

    records = []
    radius = config["evaluation"]["relaxed_radius"]
    for threshold in np.linspace(args.minimum, args.maximum, args.steps):
        clean = calculate(*clean_data, float(threshold), radius)
        occluded = calculate(*occluded_data, float(threshold), radius)
        final_score = score(clean, occluded)
        records.append(
            {
                "threshold": float(threshold),
                "final_score": final_score,
                "clean": clean,
                "occluded": occluded,
            }
        )
        print(
            f"threshold={threshold:.3f} score={final_score:.4f} "
            f"dice={clean['dice']:.4f} recall={clean['recall']:.4f} "
            f"connectivity={clean['connectivity_ratio']:.4f}",
            flush=True,
        )
    best = max(records, key=lambda record: record["final_score"])
    checkpoint["threshold"] = best["threshold"]
    checkpoint["score_calibration"] = best
    torch.save(checkpoint, checkpoint_path)
    report = {"selected": best, "all_thresholds": records}
    (checkpoint_path.parent / "threshold_calibration.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"selected_threshold={best['threshold']:.3f}", flush=True)
    print(f"validation_final_score={best['final_score']:.6f}", flush=True)


if __name__ == "__main__":
    main()
