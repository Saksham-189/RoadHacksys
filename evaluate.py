from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from roadseg.data import RoadDataset
from roadseg.metrics import evaluate_model
from roadseg.model import build_model


def make_dataset(config: dict, occluded: bool) -> RoadDataset:
    return RoadDataset(
        config["manifest"],
        "test",
        config["data"]["mean"],
        config["data"]["std"],
        training=False,
        occlusion_probability=1.0 if occluded else 0.0,
        deterministic_occlusion=occluded,
        seed=config["seed"] + 10_000,
    )


def save_examples(
    model: torch.nn.Module,
    dataset: RoadDataset,
    device: torch.device,
    threshold: float,
    destination: Path,
    count: int = 6,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    mean = dataset.mean
    std = dataset.std
    model.eval()
    for index in indices:
        item = dataset[index]
        image = item["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            probability = model(image).sigmoid()[0, 0].cpu().numpy()
        prediction = probability >= threshold
        target = item["mask"][0].numpy() > 0.5
        rgbn = item["image"].numpy() * std + mean
        rgb = np.clip(rgbn[:3].transpose(1, 2, 0), 0, 1)
        overlay = rgb.copy()
        overlay[prediction] = 0.55 * overlay[prediction] + 0.45 * np.array(
            [1.0, 0.1, 0.05]
        )

        figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
        panels = (rgb, target, probability, overlay)
        titles = ("RGB input", "OSM target", "Road probability", "Prediction overlay")
        for axis, panel, title in zip(axes, panels, titles):
            axis.imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0, vmax=1)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        figure.suptitle(
            f"{item['tile_id']} | occlusion: {item['occlusion_kind']}", fontsize=10
        )
        figure.savefig(destination / f"{item['tile_id']}.png", dpi=150)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained E012 model")
    parser.add_argument(
        "--checkpoint", default="runs/e013_pretrained_segformer_rgbn/best.pt"
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    threshold = float(checkpoint["threshold"])
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    output = checkpoint_path.parent

    results = {}
    datasets = {}
    for name, occluded in (("clean", False), ("occluded", True)):
        dataset = make_dataset(config, occluded)
        datasets[name] = dataset
        loader = DataLoader(
            dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["workers"],
            pin_memory=True,
        )
        results[name] = evaluate_model(
            model,
            loader,
            device,
            threshold,
            config["evaluation"]["relaxed_radius"],
        )
        print(f"{name}: {json.dumps(results[name], indent=2)}", flush=True)

    clean, occluded = results["clean"], results["occluded"]
    final_score = (
        0.20 * clean["iou"]
        + 0.20 * clean["dice"]
        + 0.15 * clean["recall"]
        + 0.20 * occluded["occlusion_recall"]
        + 0.15 * clean["connectivity_ratio"]
        + 0.10 * clean["relaxed_iou"]
    )
    report = {
        "model": "SegFormer-B0",
        "input_bands": "RGBN",
        "synthetic_occlusion_training": True,
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint["epoch"],
        "threshold": threshold,
        "test_tiles": len(datasets["clean"]),
        "clean": clean,
        "occluded": occluded,
        "final_score": final_score,
    }
    (output / "test_metrics.json").write_text(json.dumps(report, indent=2))
    save_examples(
        model, datasets["clean"], device, threshold, output / "examples_clean"
    )
    save_examples(
        model, datasets["occluded"], device, threshold, output / "examples_occluded"
    )
    print(f"final_score={final_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
