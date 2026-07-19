from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from roadgraph.benchmark import create_synthetic_gaps, evaluate_healing
from roadgraph.healing import EXPERIMENTS, heal_mask


def select_rows(manifest: Path, count: int) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if 0.03 <= float(row["road_pixel_ratio"]) <= 0.55
    ]
    rows.sort(key=lambda row: float(row["road_pixel_ratio"]))
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/processed/bengaluru_edge/t256_s128/tile_manifest.csv",
    )
    parser.add_argument("--tiles", type=int, default=40)
    parser.add_argument("--gaps", type=int, default=3)
    parser.add_argument("--output", default="runs/part2_healing")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = select_rows(Path(args.manifest), args.tiles)
    aggregate: dict[str, list[dict[str, float]]] = defaultdict(list)
    details = []
    bridge_details = []

    for tile_index, row in enumerate(rows):
        original = np.asarray(Image.open(row["mask_path"])) > 127
        broken, gap_target, probability = create_synthetic_gaps(
            original, seed=42 + tile_index, gap_count=args.gaps
        )
        for experiment, config in EXPERIMENTS.items():
            start = time.perf_counter()
            healed, bridges = heal_mask(broken, probability, config)
            metrics = evaluate_healing(
                original, broken, healed, gap_target, seed=4200 + tile_index
            )
            metrics["runtime_seconds"] = time.perf_counter() - start
            aggregate[experiment].append(metrics)
            details.append(
                {
                    "tile_id": row["tile_id"],
                    "experiment": experiment,
                    **metrics,
                    "bridges_added": len(bridges),
                }
            )
            for bridge in bridges:
                bridge_details.append(
                    {
                        "tile_id": row["tile_id"],
                        "experiment": experiment,
                        **bridge,
                    }
                )
        print(f"[{tile_index + 1}/{len(rows)}] {row['tile_id']}", flush=True)

    scoreboard = []
    metric_names = list(next(iter(aggregate.values()))[0])
    for experiment, records in aggregate.items():
        result = {"experiment": experiment, "tiles": len(records)}
        for metric in metric_names:
            result[metric] = float(np.mean([record[metric] for record in records]))
        scoreboard.append(result)
    scoreboard.sort(key=lambda row: row["healing_score"], reverse=True)
    for rank, row in enumerate(scoreboard, start=1):
        row["rank"] = rank

    def write_csv(path: Path, records: list[dict]) -> None:
        if not records:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv(output / "part2_healing_scoreboard.csv", scoreboard)
    write_csv(output / "tile_metrics.csv", details)
    write_csv(output / "candidate_bridges.csv", bridge_details)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "tiles": len(rows),
                "gaps_per_tile": args.gaps,
                "winner": scoreboard[0],
                "scoreboard": scoreboard,
            },
            indent=2,
        )
    )
    print(json.dumps(scoreboard, indent=2))


if __name__ == "__main__":
    main()
