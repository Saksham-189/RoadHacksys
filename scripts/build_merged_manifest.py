from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_csv_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge AOI tile manifests into one experiment manifest.")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--variant", default="t256_s128")
    parser.add_argument("--out", default="data/processed/merged_t256_s128_manifest.csv")
    parser.add_argument("--train-aois", default="")
    parser.add_argument("--val-aois", default="")
    parser.add_argument("--test-aois", default="")
    args = parser.parse_args()

    root = Path(args.processed_root)
    manifests = sorted(root.glob(f"*/{args.variant}/tile_manifest.csv"))
    if not manifests:
        raise FileNotFoundError(f"No manifests found under {root}/*/{args.variant}/tile_manifest.csv")

    frames = []
    for path in manifests:
        frame = pd.read_csv(path)
        if "aoi" not in frame.columns:
            frame["aoi"] = path.parents[1].name
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)

    train_aois = parse_csv_list(args.train_aois)
    val_aois = parse_csv_list(args.val_aois)
    test_aois = parse_csv_list(args.test_aois)

    if train_aois or val_aois or test_aois:
        def split_for_aoi(aoi: str) -> str:
            if aoi in train_aois:
                return "train"
            if aoi in val_aois:
                return "val"
            if aoi in test_aois:
                return "test"
            return "unused"

        merged["split"] = merged["aoi"].map(split_for_aoi)
        merged = merged[merged["split"] != "unused"].copy()
        split_strategy = "explicit_geographic_aoi"
    else:
        aois = sorted(merged["aoi"].unique())
        if len(aois) >= 3:
            train_aois = set(aois[:-2])
            val_aois = {aois[-2]}
            test_aois = {aois[-1]}
            merged["split"] = merged["aoi"].map(
                lambda aoi: "train" if aoi in train_aois else "val" if aoi in val_aois else "test"
            )
            split_strategy = "auto_geographic_aoi"
        else:
            split_strategy = "preserve_source_splits"

    merged["split_strategy"] = split_strategy
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    print(f"Merged manifest: {output}")
    print(merged.groupby(["aoi", "split"]).size())


if __name__ == "__main__":
    main()

