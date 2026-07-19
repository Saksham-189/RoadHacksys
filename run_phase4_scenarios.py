from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import yaml

from mobility.io import write_csv
from mobility.scenario import ScenarioSpec
from mobility.simulation import SimulationEngine


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def edge_action(action: str, row: dict, **extra) -> dict:
    return {
        "action": action,
        "edges": [
            {"source": row["source"], "target": row["target"], **extra}
        ],
    }


def build_presets(config: dict, graph: nx.Graph) -> list[dict]:
    paths = config["paths"]
    nodes = read_rows(paths["node_criticality"])
    edges = read_rows(paths["edge_criticality"])
    flow = read_rows(paths["baseline_edge_flow"])
    betweenness = read_rows(paths["edge_betweenness"])
    bridges = [row for row in betweenness if row["graph_bridge"].lower() == "true"]
    top_nodes = [str(row["node_id"]) for row in nodes[:3]]
    top_edges = edges[:3]
    centre = graph.nodes[top_nodes[0]]
    x, y = float(centre["x"]), float(centre["y"])
    half = 500.0
    square = {
        "type": "Polygon",
        "coordinates": [
            [
                [x - half, y - half],
                [x + half, y - half],
                [x + half, y + half],
                [x - half, y + half],
                [x - half, y - half],
            ]
        ],
    }
    randomizer = random.Random(int(config["seed"]))
    random_nodes = randomizer.sample(sorted(map(str, graph.nodes)), len(top_nodes))
    return [
        {
            "scenario_id": "D001",
            "name": "Flood 250 m around top flow-critical junction",
            "hazard_type": "flood",
            "actions": [
                {"action": "close_circle", "x": x, "y": y, "radius_m": 250}
            ],
        },
        {
            "scenario_id": "D002",
            "name": "Compound 500 m and 1 km flood zones",
            "hazard_type": "flood",
            "actions": [
                {"action": "close_circle", "x": x, "y": y, "radius_m": 500},
                {
                    "action": "close_circle",
                    "x": float(graph.nodes[top_nodes[1]]["x"]),
                    "y": float(graph.nodes[top_nodes[1]]["y"]),
                    "radius_m": 1000,
                },
            ],
        },
        {
            "scenario_id": "D003",
            "name": "Accident closes top flow-critical edge",
            "hazard_type": "accident",
            "actions": [edge_action("close_edges", edges[0])],
        },
        {
            "scenario_id": "D004",
            "name": "Construction halves capacity on top V/C edge",
            "hazard_type": "construction",
            "actions": [
                edge_action(
                    "capacity_derating", flow[0], capacity_factor=0.5
                )
            ],
        },
        {
            "scenario_id": "D005",
            "name": "Highest-betweenness graph bridge unavailable",
            "hazard_type": "bridge_closure",
            "actions": [edge_action("close_edges", bridges[0])],
        },
        {
            "scenario_id": "D006",
            "name": "One-kilometre critical-sector closure",
            "hazard_type": "sector_disaster",
            "actions": [
                {
                    "action": "close_polygon",
                    "geometry": square,
                    "crs": graph.graph.get("crs", "EPSG:32643"),
                }
            ],
        },
        {
            "scenario_id": "D007",
            "name": "Top three critical junctions unavailable",
            "hazard_type": "compound_node_failure",
            "actions": [{"action": "close_nodes", "node_ids": top_nodes}],
        },
        {
            "scenario_id": "D008",
            "name": "Top three critical road links unavailable",
            "hazard_type": "compound_edge_failure",
            "actions": [
                {
                    "action": "close_edges",
                    "edges": [
                        {"source": row["source"], "target": row["target"]}
                        for row in top_edges
                    ],
                }
            ],
        },
        {
            "scenario_id": "D009",
            "name": "Three matched random junction closures",
            "hazard_type": "random_failure",
            "actions": [{"action": "close_nodes", "node_ids": random_nodes}],
        },
    ]


def scoreboard_row(preview, exact) -> dict:
    summary = exact.summary
    return {
        "scenario_id": summary["scenario_id"],
        "hazard_type": summary["hazard_type"],
        "closed_nodes": summary["closed_nodes"],
        "closed_edges": summary["closed_edges"],
        "degraded_edges": summary["degraded_edges"],
        "served_demand_ratio": summary["served_demand_ratio"],
        "path_resilience": summary["path_resilience"],
        "service_adjusted_resilience": summary[
            "service_adjusted_resilience"
        ],
        "resilience_band": summary["resilience_band"],
        "affected_demand_ratio": summary["affected_demand_ratio"],
        "path_length_increase": summary["path_length_increase"],
        "travel_time_increase": summary["travel_time_increase"],
        "efficiency_loss": summary["global_efficiency_loss"],
        "largest_component_loss": summary["largest_component_loss"],
        "newly_overloaded_edges": summary["newly_overloaded_edges"],
        "preview_runtime_seconds": preview.runtime_seconds,
        "exact_runtime_seconds": exact.runtime_seconds,
        "msa_iterations": summary["msa_iterations"],
        "msa_relative_gap": summary["msa_relative_gap"],
        "preview_resilience_error": abs(
            preview.summary["service_adjusted_resilience"]
            - summary["service_adjusted_resilience"]
        ),
        "preview_served_demand_error": abs(
            preview.summary["served_demand_ratio"]
            - summary["served_demand_ratio"]
        ),
    }


def plot_results(rows: list[dict], output: Path) -> None:
    labels = [row["scenario_id"] for row in rows]
    values = [float(row["service_adjusted_resilience"]) for row in rows]
    colors = [plt.cm.RdYlGn(value) for value in values]
    fig, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, values, color=colors)
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    axis.axhline(0.8, color="#238636", linestyle="--", linewidth=1)
    axis.axhline(0.6, color="#d29922", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Service-adjusted resilience")
    axis.set_xlabel("Disruption scenario")
    axis.set_title("Urban Road-Network Resilience Under Disruption")
    fig.tight_layout()
    fig.savefig(
        output / "resilience_comparison.png", dpi=180, facecolor="white"
    )
    plt.close(fig)

    worst = min(rows, key=lambda row: float(row["service_adjusted_resilience"]))
    zones_path = output / "scenarios" / worst["scenario_id"] / "affected_zones.geojson"
    zones = json.loads(zones_path.read_text())
    fig, axis = plt.subplots(figsize=(8, 7))
    impacts = [
        float(feature["properties"]["impact_score"])
        for feature in zones["features"]
    ]
    maximum_impact = max(impacts, default=1.0)
    for feature in zones["features"]:
        polygon = feature["geometry"]["coordinates"][0]
        xs, ys = zip(*polygon)
        impact = float(feature["properties"]["impact_score"])
        axis.fill(
            xs,
            ys,
            color=plt.cm.YlOrRd(impact / max(maximum_impact, 1e-12)),
            edgecolor="#7f1d1d",
            linewidth=0.15,
            alpha=0.85,
        )
    axis.set_aspect("equal")
    axis.set_title(f"Affected mobility zones: {worst['scenario_id']}")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(output / "scenario_impact_map.png", dpi=180, facecolor="white")
    plt.close(fig)


def write_report(rows: list[dict], output: Path) -> None:
    worst = min(rows, key=lambda row: float(row["service_adjusted_resilience"]))
    best = max(rows, key=lambda row: float(row["service_adjusted_resilience"]))
    lines = [
        "# Phase 4 Implementation Report",
        "",
        "## Purpose",
        "",
        "This phase tests how the inferred urban road network responds when "
        "junctions, links, bridges, or whole geographic sectors become unavailable. "
        "Traffic demand and capacity are relative satellite-derived estimates, not "
        "observed vehicle counts.",
        "",
        "## Method",
        "",
        "1. Freeze the Part 3 transport graph and gravity-demand matrix.",
        "2. Resolve each JSON scenario without mutating the baseline graph.",
        "3. Produce a rapid all-or-nothing preview.",
        "4. Reassign the same demand with converged MSA and BPR congestion costs.",
        "5. Compare paired baseline/disrupted routes and map affected 500 m zones.",
        "6. Rank scenarios by service-adjusted resilience.",
        "",
        "## Results",
        "",
        "| Scenario | Disconnected | Affected demand | Path change | Time change | Resilience | Band |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | "
            f"{100 * (1 - float(row['served_demand_ratio'])):.2f}% | "
            f"{100 * float(row['affected_demand_ratio']):.2f}% | "
            f"{100 * float(row['path_length_increase'] or 0):.2f}% | "
            f"{100 * float(row['travel_time_increase'] or 0):.2f}% | "
            f"{float(row['service_adjusted_resilience']):.3f} | "
            f"{row['resilience_band']} |"
        )
    lines.extend(
        [
            "",
            "## Main Finding",
            "",
            f"`{worst['scenario_id']}` is the most damaging tested case, with "
            f"{100 * (1 - float(worst['served_demand_ratio'])):.2f}% of estimated "
            "mobility demand disconnected and an official resilience of "
            f"{float(worst['service_adjusted_resilience']):.3f}. "
            f"`{best['scenario_id']}` produces the least degradation among the "
            "non-empty preset disruptions.",
            "",
            f"In plain language: under `{worst['scenario_id']}`, "
            f"{100 * float(worst['affected_demand_ratio']):.2f}% of estimated "
            "mobility demand is affected. The surviving paired routes show a "
            f"{100 * float(worst['path_length_increase'] or 0):.2f}% aggregate "
            "path-length increase and a "
            f"{100 * float(worst['travel_time_increase'] or 0):.2f}% aggregate "
            "travel-time increase. The affected-zone GeoJSON identifies the "
            "specific 500 m cells where those origins and destinations lie.",
            "",
            "For every scenario, the scenario folder contains OD-level impacts, "
            "closed infrastructure, rerouting flow changes, affected zones, and ten "
            "high-impact route examples. These outputs directly answer which areas "
            "lose access, how far surviving journeys detour, and where rerouted "
            "traffic creates additional burden.",
            "",
            "## Interpretation",
            "",
            "The official score multiplies the fraction of demand still served by "
            "the canonical shortest-path ratio. A low score therefore captures both "
            "disconnection and long detours. Results should be presented as "
            "predictive relative stress tests rather than calibrated traffic forecasts.",
        ]
    )
    (output / "PHASE4_IMPLEMENTATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 preset scenarios.")
    parser.add_argument("--config", default="configs/phase4.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    engine = SimulationEngine(config)
    scenarios = build_presets(config, engine.graph)
    if args.limit:
        scenarios = scenarios[: args.limit]
    scenario_config_dir = Path("configs/scenarios")
    scenario_config_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for data in scenarios:
        spec = ScenarioSpec.from_dict(data)
        (scenario_config_dir / f"{spec.scenario_id}.json").write_text(
            json.dumps(spec.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"Running {spec.scenario_id}: preview")
        preview = engine.preview(spec)
        destination = engine.output / "scenarios" / spec.scenario_id
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "preview_summary.json").write_text(
            json.dumps(preview.summary, indent=2), encoding="utf-8"
        )
        print(f"Running {spec.scenario_id}: exact")
        exact = engine.simulate(spec)
        exact.to_directory(destination)
        rows.append(scoreboard_row(preview, exact))
    write_csv(engine.output / "scenario_scoreboard.csv", rows)
    plot_results(rows, engine.output)
    write_report(rows, engine.output)
    print(f"Completed {len(rows)} scenarios in {engine.output.resolve()}")


if __name__ == "__main__":
    main()
