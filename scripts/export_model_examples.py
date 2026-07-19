from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadseg.engine import load_checkpoint, make_device, make_loader
from roadseg.metrics import compute_scores, empty_totals, update_metrics


def overlay_image(image: np.ndarray, target: np.ndarray, pred: np.ndarray) -> Image.Image:
    rgb = np.moveaxis(image[:3], 0, -1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    overlay = rgb.copy()
    truth = target > 0.5
    prediction = pred > 0
    overlay[truth] = (255, 80, 40)
    overlay[prediction] = (40, 170, 255)
    overlay[np.logical_and(truth, prediction)] = (80, 235, 80)
    return Image.fromarray((0.55 * rgb + 0.45 * overlay).astype(np.uint8), mode="RGB")


def add_label(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 34), "white")
    canvas.paste(image, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), text, fill=(20, 20, 20))
    return canvas


def make_grid(items: list[tuple[str, Image.Image]], path: Path) -> None:
    if not items:
        return
    cell_w = max(img.width for _, img in items)
    cell_h = max(img.height for _, img in items)
    cols = min(3, len(items))
    rows = int(np.ceil(len(items) / cols))
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, (_, image) in enumerate(items):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        grid.paste(image, (x, y))
    grid.save(path)


@torch.no_grad()
def collect_examples(checkpoint: str, split: str, occluded: bool, output_dir: Path, limit: int = 4) -> None:
    device = make_device("auto")
    model, cfg, _ = load_checkpoint(checkpoint, device)
    model.eval()
    loader = make_loader(cfg, split, train=False, occluded_eval=occluded)

    rows = []
    examples = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        for idx in range(images.size(0)):
            totals = update_metrics(empty_totals(), logits[idx : idx + 1], masks[idx : idx + 1])
            scores = compute_scores(totals)
            tile_id = batch["tile_id"][idx]
            rows.append({"tile_id": tile_id, **scores})
            image_np = images[idx].detach().cpu().numpy()
            target_np = masks[idx, 0].detach().cpu().numpy()
            pred_np = preds[idx, 0].detach().cpu().numpy()
            examples.append((tile_id, scores, overlay_image(image_np, target_np, pred_np)))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / ("tile_metrics_occluded.csv" if occluded else "tile_metrics_clean.csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    positive = [item for item in examples if item[1]["dice"] > 0]
    top = sorted(positive, key=lambda item: item[1]["dice"], reverse=True)[:limit]
    low = sorted(positive, key=lambda item: item[1]["dice"])[:limit]
    middle = sorted(positive, key=lambda item: abs(item[1]["dice"] - np.median([x[1]["dice"] for x in positive])))[:limit]

    prefix = "occluded" if occluded else "clean"
    selected_groups = {"strong": top, "typical": middle, "failure": low}
    for group, selected in selected_groups.items():
        labelled = []
        for tile_id, scores, img in selected:
            label = f"{tile_id} | Dice {scores['dice']:.3f} IoU {scores['iou']:.3f}"
            labelled_img = add_label(img, label)
            labelled_img.save(output_dir / f"{prefix}_{group}_{tile_id}.png")
            labelled.append((tile_id, labelled_img))
        make_grid(labelled, output_dir / f"{prefix}_{group}_grid.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export strong/typical/failure visual examples for a model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    collect_examples(args.checkpoint, args.split, occluded=False, output_dir=out)
    collect_examples(args.checkpoint, args.split, occluded=True, output_dir=out)
    print(out)


if __name__ == "__main__":
    main()
