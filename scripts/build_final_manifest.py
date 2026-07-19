from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    original = read_rows(Path("data/processed/merged_t256_s128_manifest.csv"))
    expanded = read_rows(Path("data/expanded_v2/expanded_manifest.csv"))
    rows = original + expanded
    tile_ids = [row["tile_id"] for row in rows]
    if len(tile_ids) != len(set(tile_ids)):
        raise RuntimeError("Duplicate tile IDs in combined manifest")
    destination = Path("data/expanded_v2/final_manifest.csv")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {destination}")
    print("Splits:", dict(Counter(row["split"] for row in rows)))
    print("AOIs:", len(set(row["aoi"] for row in rows)))


if __name__ == "__main__":
    main()
