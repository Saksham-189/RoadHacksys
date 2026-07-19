from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from roadseg.data import RoadDataset
from roadseg.legacy_metrics import (
    MetricTotals,
    compute_scores,
    final_score,
    update_metrics,
)
from roadseg.legacy_occlusion import apply_legacy_occlusion
from roadseg.model import build_model


class LegacyOccludedDataset(Dataset):
    def __init__(self, clean: RoadDataset, mean: list[float], std: list[float], seed: int):
        self.clean = clean
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
        self.seed = seed

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, index: int) -> dict:
        item = self.clean[index]
        raw = item["image"].numpy() * self.std + self.mean
        raw, occlusion = apply_legacy_occlusion(
            raw, random.Random(self.seed + index)
        )
        item["image"] = torch.from_numpy(
            np.ascontiguousarray((raw - self.mean) / self.std)
        ).float()
        item["occlusion_mask"] = torch.from_numpy(
            np.ascontiguousarray(occlusion[None])
        ).float()
        return item


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    include_occlusion: bool,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals = MetricTotals()
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        occlusions = batch["occlusion_mask"].to(device) if include_occlusion else None
        update_metrics(
            totals,
            model(images),
            targets,
            occlusions,
            threshold=0.5,
            relaxed_buffer=2,
        )
    return compute_scores(totals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="runs/e013_pretrained_segformer_rgbn/best.pt"
    )
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    data = config["data"]
    clean_dataset = RoadDataset(
        config["manifest"],
        "test",
        data["mean"],
        data["std"],
        training=False,
        occlusion_probability=0.0,
        seed=config["seed"],
    )
    occluded_dataset = LegacyOccludedDataset(
        clean_dataset, data["mean"], data["std"], config["seed"]
    )
    clean = evaluate(
        model, clean_dataset, device, data["batch_size"], include_occlusion=False
    )
    occluded = evaluate(
        model, occluded_dataset, device, data["batch_size"], include_occlusion=True
    )
    result = {
        "protocol": "legacy_v1_exact",
        "threshold": 0.5,
        "test_tiles": len(clean_dataset),
        "clean": clean,
        "occluded": occluded,
        "final_score": final_score(clean, occluded),
    }
    destination = checkpoint_path.parent / "legacy_v1_test_metrics.json"
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    scoreboard = Path("runs/legacy_v1_scoreboard.csv")
    historical = {
        "experiment": "E012_historical",
        "model": "SegFormer RGBN synthetic occlusion",
        "threshold": 0.5,
        "iou": 0.20579830165245788,
        "dice": 0.34134780480355037,
        "precision": 0.21568843656022912,
        "recall": 0.817788917709961,
        "occlusion_recall": 0.9179072016000727,
        "relaxed_iou": 0.27010085972786546,
        "connectivity_ratio": 0.8711244557436052,
        "components_count": 5.4375,
        "final_score": 0.5733575793398767,
        "source": "recovered original held-out evaluation",
    }
    current = {
        "experiment": "E013_pretrained",
        "model": "Pretrained SegFormer-B0 RGBN synthetic occlusion",
        "threshold": 0.5,
        "iou": clean["iou"],
        "dice": clean["dice"],
        "precision": clean["precision"],
        "recall": clean["recall"],
        "occlusion_recall": occluded["occlusion_recall"],
        "relaxed_iou": clean["relaxed_iou"],
        "connectivity_ratio": clean["connectivity_ratio"],
        "components_count": clean["components_count"],
        "final_score": result["final_score"],
        "source": str(destination),
    }
    rows = [historical, current]
    with scoreboard.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(historical))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["final_score"], reverse=True))


if __name__ == "__main__":
    main()
