from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from roadseg.config import resolve_path


SCOREBOARD_COLUMNS = [
    "exp_id",
    "model",
    "family",
    "backbone",
    "input_bands",
    "train_aois",
    "test_aois",
    "epochs",
    "best_epoch",
    "params_m",
    "iou",
    "dice",
    "precision",
    "recall",
    "occlusion_recall",
    "relaxed_iou",
    "connectivity_ratio",
    "components_count",
    "base_score",
    "final_score",
    "rank",
    "checkpoint_path",
    "notes",
]


def ensure_scoreboard(path: str | Path = "runs/model_scoreboard.csv") -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCOREBOARD_COLUMNS)
            writer.writeheader()
    return path


def upsert_scoreboard(row: dict[str, Any], path: str | Path = "runs/model_scoreboard.csv") -> Path:
    path = ensure_scoreboard(path)
    df = pd.read_csv(path)
    if df.empty:
        df = pd.DataFrame(columns=SCOREBOARD_COLUMNS)
    df = df.astype("object")

    clean_row = {col: row.get(col, "") for col in SCOREBOARD_COLUMNS}
    exp_id = str(clean_row["exp_id"])
    existing = df["exp_id"].astype(str) == exp_id if "exp_id" in df.columns else []
    if len(df) and existing.any():
        idx = df.index[existing][0]
        for key, value in clean_row.items():
            if value != "":
                df.at[idx, key] = value
    else:
        df = pd.concat([df, pd.DataFrame([clean_row])], ignore_index=True)

    metric_cols = ["iou", "dice", "recall", "occlusion_recall", "connectivity_ratio", "relaxed_iou"]
    for col in metric_cols:
        if col not in df.columns:
            df[col] = ""
    numeric_metrics = {col: pd.to_numeric(df[col], errors="coerce") for col in metric_cols}
    base_ready = (
        numeric_metrics["iou"].notna()
        & numeric_metrics["dice"].notna()
        & numeric_metrics["recall"].notna()
        & numeric_metrics["connectivity_ratio"].notna()
        & numeric_metrics["relaxed_iou"].notna()
    )
    final_ready = base_ready & numeric_metrics["occlusion_recall"].notna()
    df.loc[base_ready, "base_score"] = (
        0.25 * numeric_metrics["iou"]
        + 0.25 * numeric_metrics["dice"]
        + 0.20 * numeric_metrics["recall"]
        + 0.20 * numeric_metrics["connectivity_ratio"]
        + 0.10 * numeric_metrics["relaxed_iou"]
    ).round(6).astype(str)
    df.loc[final_ready, "final_score"] = (
        0.20 * numeric_metrics["iou"]
        + 0.20 * numeric_metrics["dice"]
        + 0.15 * numeric_metrics["recall"]
        + 0.20 * numeric_metrics["occlusion_recall"]
        + 0.15 * numeric_metrics["connectivity_ratio"]
        + 0.10 * numeric_metrics["relaxed_iou"]
    ).round(6).astype(str)

    if "final_score" in df.columns:
        numeric = pd.to_numeric(df["final_score"], errors="coerce")
        ranked = numeric.rank(method="min", ascending=False)
        df["rank"] = ranked.astype("Int64").astype(str).replace("<NA>", "")

    df.to_csv(path, index=False)
    return path
