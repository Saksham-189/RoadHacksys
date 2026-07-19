from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadgraph.benchmark import create_synthetic_gaps
from roadgraph.healing import EXPERIMENTS, heal_mask


def panel(
    base: np.ndarray,
    title: str,
    added: np.ndarray | None = None,
    gap: np.ndarray | None = None,
) -> Image.Image:
    rgb = np.zeros((*base.shape, 3), dtype=np.uint8)
    rgb[base] = (225, 225, 225)
    if gap is not None:
        rgb[gap] = (255, 170, 30)
    if added is not None:
        rgb[added] = (40, 220, 110)
    image = Image.fromarray(rgb)
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((8, 8), title, fill=(20, 20, 20))
    return canvas


def main() -> None:
    manifest = Path(
        "data/processed/bengaluru_edge/t256_s128/tile_manifest.csv"
    )
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = min(rows, key=lambda item: abs(float(item["road_pixel_ratio"]) - 0.22))
    original = np.asarray(Image.open(row["mask_path"])) > 127
    broken, gap, probability = create_synthetic_gaps(original, 42, gap_count=3)
    b003, _ = heal_mask(broken, probability, EXPERIMENTS["B003"])
    b007, _ = heal_mask(broken, probability, EXPERIMENTS["B007"])
    probability_panel = panel(
        probability > 0.25, "Probability evidence", gap=gap
    )
    panels = [
        panel(original, "Original OSM road"),
        panel(broken, "Controlled gaps", gap=gap),
        panel(b003, "B003 additions", added=b003 & ~broken),
        panel(b007, "B007 additions", added=b007 & ~broken),
        probability_panel,
    ]
    grid = Image.new(
        "RGB",
        (sum(image.width for image in panels), max(image.height for image in panels)),
        "white",
    )
    left = 0
    for image in panels:
        grid.paste(image, (left, 0))
        left += image.width
    output = Path("runs/part2_healing/part2_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    print(output)


if __name__ == "__main__":
    main()
