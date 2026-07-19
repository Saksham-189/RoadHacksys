from __future__ import annotations

import csv
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="data/expanded_v2/final_manifest.csv"
    )
    args = parser.parse_args()
    manifest = Path(args.manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tile_ids = [row["tile_id"] for row in rows]
    assert len(tile_ids) == len(set(tile_ids)), "Duplicate tile IDs"
    for row in rows:
        image_path = Path(row["image_rgbn_path"])
        mask_path = Path(row["mask_path"])
        assert image_path.exists(), image_path
        assert mask_path.exists(), mask_path
        image = np.load(image_path, mmap_mode="r")
        mask = np.asarray(Image.open(mask_path))
        assert image.shape == (4, 256, 256), (image_path, image.shape)
        assert mask.shape == (256, 256), (mask_path, mask.shape)
        assert set(np.unique(mask)).issubset({0, 255}), mask_path
    print(f"Validated {len(rows)} tiles")
    print(f"Splits: {dict(Counter(row['split'] for row in rows))}")
    print(f"AOIs: {dict(Counter(row['aoi'] for row in rows))}")


if __name__ == "__main__":
    main()
