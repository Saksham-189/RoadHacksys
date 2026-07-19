from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    scoreboard = Path("runs/model_scoreboard.csv")
    if not scoreboard.exists():
        raise FileNotFoundError(scoreboard)
    df = pd.read_csv(scoreboard)
    df = df[~df["exp_id"].astype(str).str.startswith("SMOKE")].copy()
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    cols = [
        "rank",
        "exp_id",
        "model",
        "input_bands",
        "iou",
        "dice",
        "recall",
        "occlusion_recall",
        "connectivity_ratio",
        "base_score",
        "final_score",
        "checkpoint_path",
    ]

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    df[cols].to_csv(report_dir / "final_model_ranking.csv", index=False)

    lines = [
        "# Part 1 Final Model Ranking",
        "",
        "Smoke runs are excluded. Scores are computed on the held-out `bengaluru_edge` test AOI.",
        "",
        to_markdown_table(df[cols[:-1]]),
        "",
        "## Winner",
        "",
    ]
    if not df.empty:
        winner = df.iloc[0]
        lines.extend(
            [
                f"- Model: {winner['model']}",
                f"- Experiment: {winner['exp_id']}",
                f"- Input bands: {winner['input_bands']}",
                f"- Final score: {winner['final_score']:.6f}",
                f"- Checkpoint: `{winner['checkpoint_path']}`",
            ]
        )
    (report_dir / "final_model_ranking.md").write_text("\n".join(lines), encoding="utf-8")
    print(report_dir / "final_model_ranking.csv")
    print(report_dir / "final_model_ranking.md")


if __name__ == "__main__":
    main()
