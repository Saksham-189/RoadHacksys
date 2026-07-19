from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np

from mobility.assignment import edge_key


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def base_segments(graph: nx.Graph) -> tuple[list, np.ndarray]:
    segments = []
    for first, second in graph.edges:
        segments.append(
            [
                (float(graph.nodes[first]["x"]), float(graph.nodes[first]["y"])),
                (float(graph.nodes[second]["x"]), float(graph.nodes[second]["y"])),
            ]
        )
    return segments, np.asarray(segments)


def save_resilience(output: Path) -> None:
    rows = read_rows(output / "resilience_curves.csv")
    figure, axis = plt.subplots(figsize=(8, 5))
    for strategy in sorted({row["strategy"] for row in rows}):
        selected = sorted(
            (row for row in rows if row["strategy"] == strategy),
            key=lambda row: float(row["removal_percentage"]),
        )
        axis.plot(
            [100 * float(row["removal_percentage"]) for row in selected],
            [float(row["resilience_index"]) for row in selected],
            marker="o",
            label=strategy.replace("_", " ").title(),
        )
    axis.set(
        xlabel="Nodes removed (%)",
        ylabel="Resilience Index",
        title="Urban Network Stress Test",
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "resilience_curves.png", dpi=180)
    plt.close(figure)


def save_flow_map(graph: nx.Graph, output: Path) -> None:
    rows = read_rows(output / "edge_baseline_flow.csv")
    values = {
        edge_key(row["source"], row["target"]): float(
            row["volume_capacity_ratio"]
        )
        for row in rows
    }
    segments, _ = base_segments(graph)
    colors = [values.get(edge_key(first, second), 0) for first, second in graph.edges]
    figure, axis = plt.subplots(figsize=(10, 8))
    collection = LineCollection(
        segments,
        array=np.asarray(colors),
        cmap="RdYlGn_r",
        linewidths=0.8,
        clim=(0, min(1.5, max(colors))),
    )
    axis.add_collection(collection)
    axis.autoscale()
    axis.set_title("Relative Traffic Load (Volume / Capacity)")
    axis.set_aspect("equal")
    axis.axis("off")
    figure.colorbar(collection, ax=axis, fraction=0.03, label="V/C ratio")
    figure.tight_layout()
    figure.savefig(output / "relative_flow_map.png", dpi=180)
    plt.close(figure)


def save_criticality_map(graph: nx.Graph, output: Path) -> None:
    rows = read_rows(output / "node_ablation_results.csv")
    scores = {row["node_id"]: float(row["flow_criticality"]) for row in rows}
    segments, _ = base_segments(graph)
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.add_collection(
        LineCollection(segments, colors="#bac2cc", linewidths=0.45, alpha=0.8)
    )
    nodes = [node for node in graph if str(node) in scores]
    scatter = axis.scatter(
        [float(graph.nodes[node]["x"]) for node in nodes],
        [float(graph.nodes[node]["y"]) for node in nodes],
        c=[scores[str(node)] for node in nodes],
        cmap="YlOrRd",
        s=18,
        edgecolors="black",
        linewidths=0.2,
        zorder=3,
    )
    axis.autoscale()
    axis.set_title("Flow-Aware Gatekeeper Nodes")
    axis.set_aspect("equal")
    axis.axis("off")
    figure.colorbar(scatter, ax=axis, fraction=0.03, label="Criticality")
    figure.tight_layout()
    figure.savefig(output / "node_criticality_map.png", dpi=180)
    plt.close(figure)


def main() -> None:
    output = Path("runs/part3")
    graph = nx.read_graphml(output / "transport_graph.graphml")
    save_resilience(output)
    save_flow_map(graph, output)
    save_criticality_map(graph, output)
    print(output)


if __name__ == "__main__":
    main()
