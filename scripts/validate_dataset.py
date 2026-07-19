from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate road segmentation tile manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = pd.read_csv(args.manifest)
    required = ["tile_id", "split", "image_rgb_path", "image_rgbn_path", "mask_path"]
    missing = [col for col in required if col not in manifest.columns]
    if missing:
        raise AssertionError(f"Missing columns: {missing}")
    if manifest["tile_id"].duplicated().any():
        raise AssertionError("Duplicate tile_id values found.")

    split_counts = manifest["split"].value_counts().to_dict()
    for split in ("train", "val", "test"):
        if split not in split_counts:
            raise AssertionError(f"Missing split: {split}")

    checked = 0
    for row in manifest.itertuples(index=False):
        rgb_path = root / row.image_rgb_path
        rgbn_path = root / row.image_rgbn_path
        mask_path = root / row.mask_path
        for path in (rgb_path, rgbn_path, mask_path):
            if not path.exists():
                raise AssertionError(f"Missing file: {path}")

        rgb = Image.open(rgb_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        rgbn = np.load(rgbn_path)
        if rgb.size != mask.size:
            raise AssertionError(f"Image/mask size mismatch for {row.tile_id}: {rgb.size} vs {mask.size}")
        if rgbn.shape[0] != 4 or rgbn.shape[1:] != (rgb.size[1], rgb.size[0]):
            raise AssertionError(f"RGBN shape mismatch for {row.tile_id}: {rgbn.shape}")
        unique = set(np.unique(np.array(mask)).tolist())
        if not unique.issubset({0, 255}):
            raise AssertionError(f"Mask is not binary for {row.tile_id}: {sorted(unique)[:10]}")
        checked += 1

    print(f"Validated {checked} samples")
    print(split_counts)


if __name__ == "__main__":
    main()

