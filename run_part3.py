from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from mobility.assignment import assign_msa, edge_key
from mobility.centrality import betweenness_reference, degree_baseline
from mobility.config import load_config
from mobility.demand import generate_demands
from mobility.io import graph_to_geojson, write_csv
from mobility.stress import (
    assignment_parameters,
    build_ablation_candidates,
    run_ablations,
    run_flood_scenarios,
    run_progressive_scenarios,
)
from mobility.transport import prepare_transport_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part3_flow.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["paths"]["part3_output"])
    output.mkdir(parents=True, exist_ok=True)

    graph, preparation = prepare_transport_graph(config)
    degree_rows = degree_baseline(graph)
    node_betweenness, edge_betweenness = betweenness_reference(
        graph,
        int(config["centrality"]["betweenness_sources"]),
        int(config["seed"]),
    )
    write_csv(output / "node_degree_baseline.csv", degree_rows)
    write_csv(output / "node_betweenness_reference.csv", node_betweenness)
    write_csv(output / "edge_betweenness_reference.csv", edge_betweenness)

    gravity, uniform, demand_summary = generate_demands(graph, config, output)
    parameters = assignment_parameters(config)
    gravity_baseline = assign_msa(graph, gravity, **parameters)
    uniform_baseline = assign_msa(graph, uniform, **parameters)

    edge_flow_rows = []
    for first, second, data in graph.edges(data=True):
        key = edge_key(first, second)
        flow = gravity_baseline.edge_flows.get(key, 0.0)
        capacity = float(data["relative_capacity"])
        edge_flow_rows.append(
            {
                "source": str(first),
                "target": str(second),
                "length_m": float(data["length_m"]),
                "road_class": data["road_class"],
                "relative_capacity": capacity,
                "speed_kmh": float(data["speed_kmh"]),
                "free_flow_time_min": float(data["free_flow_time_min"]),
                "assigned_flow": flow,
                "volume_capacity_ratio": flow / max(capacity, 1e-12),
                "congested_time_min": gravity_baseline.edge_costs.get(
                    key, float(data["free_flow_time_min"])
                ),
                "overloaded": flow / max(capacity, 1e-12) > 1.0,
            }
        )
    edge_flow_rows.sort(
        key=lambda row: row["volume_capacity_ratio"], reverse=True
    )
    for rank, row in enumerate(edge_flow_rows, start=1):
        row["flow_rank"] = rank
    write_csv(output / "edge_baseline_flow.csv", edge_flow_rows)

    node_candidates, edge_candidates, throughput = build_ablation_candidates(
        graph,
        gravity_baseline,
        degree_rows,
        node_betweenness,
        edge_betweenness,
        int(config["stress"]["candidate_count"]),
    )
    node_ablation, edge_ablation = run_ablations(
        graph,
        gravity,
        gravity_baseline,
        node_candidates,
        edge_candidates,
        config,
    )
    write_csv(output / "node_ablation_results.csv", node_ablation)
    write_csv(output / "edge_ablation_results.csv", edge_ablation)

    degree_top = {
        row["node_id"]
        for row in degree_rows[
            : int(config["centrality"]["degree_ablation_count"])
        ]
    }
    write_csv(
        output / "degree_ablation_results.csv",
        [row for row in node_ablation if row["node_id"] in degree_top],
    )
    resilience = run_progressive_scenarios(
        graph,
        gravity,
        gravity_baseline,
        degree_rows,
        node_betweenness,
        node_ablation,
        throughput,
        config,
    )
    floods = run_flood_scenarios(
        graph, gravity, gravity_baseline, node_ablation, config
    )
    write_csv(output / "resilience_curves.csv", resilience)
    write_csv(output / "flood_scenarios.csv", floods)
    worst_flood = min(floods, key=lambda row: row["resilience_index"])

    node_properties = {
        row["node_id"]: {
            "flow_criticality": row["flow_criticality"],
            "resilience_if_removed": row["resilience_index"],
        }
        for row in node_ablation
    }
    edge_properties = {
        edge_key(row["source"], row["target"]): {
            "flow_criticality": row["flow_criticality"],
            "resilience_if_removed": row["resilience_index"],
        }
        for row in edge_ablation
    }
    graph_to_geojson(
        graph,
        output / "node_criticality.geojson",
        node_properties=node_properties,
    )
    graph_to_geojson(
        graph,
        output / "edge_criticality.geojson",
        edge_properties=edge_properties,
    )

    comparison = [
        {
            "experiment": "C001",
            "method": "Degree centrality baseline",
            "top_candidate": degree_rows[0]["node_id"],
            "primary_score": degree_rows[0]["degree_score"],
        },
        {
            "experiment": "C002",
            "method": "Betweenness reference",
            "top_candidate": node_betweenness[0]["node_id"],
            "primary_score": node_betweenness[0]["betweenness"],
        },
        {
            "experiment": "C010",
            "method": "Gravity-demand MSA assignment",
            "top_candidate": edge_flow_rows[0]["source"]
            + "-"
            + edge_flow_rows[0]["target"],
            "primary_score": edge_flow_rows[0]["volume_capacity_ratio"],
        },
        {
            "experiment": "C011",
            "method": "Flow-aware ablation",
            "top_candidate": node_ablation[0]["node_id"],
            "primary_score": node_ablation[0]["flow_criticality"],
        },
        {
            "experiment": "C012",
            "method": "Geographic and cascading stress tests",
            "top_candidate": worst_flood["center_node"],
            "primary_score": worst_flood["resilience_index"],
        },
    ]
    write_csv(output / "part3_experiment_comparison.csv", comparison)
    consolidation_summary = json.loads(
        (
            Path(config["paths"]["consolidation_output"])
            / "consolidation_summary.json"
        ).read_text()
    )
    summary = {
        "statement": (
            "All traffic and capacity values are relative estimates, not "
            "calibrated vehicle counts."
        ),
        "consolidation": consolidation_summary,
        "graph_preparation": preparation,
        "demand": demand_summary,
        "gravity_baseline": {
            "served_demand_ratio": gravity_baseline.served_demand_ratio,
            "mean_travel_time_min": gravity_baseline.mean_travel_time,
            "global_efficiency": gravity_baseline.global_efficiency,
            "overloaded_edges": gravity_baseline.overloaded_edges,
            "iterations": gravity_baseline.iterations,
            "convergence": gravity_baseline.convergence,
            "converged": gravity_baseline.convergence
            <= float(config["assignment"]["tolerance"]),
        },
        "uniform_sensitivity": {
            "served_demand_ratio": uniform_baseline.served_demand_ratio,
            "mean_travel_time_min": uniform_baseline.mean_travel_time,
            "global_efficiency": uniform_baseline.global_efficiency,
            "overloaded_edges": uniform_baseline.overloaded_edges,
            "iterations": uniform_baseline.iterations,
            "convergence": uniform_baseline.convergence,
            "converged": uniform_baseline.convergence
            <= float(config["assignment"]["tolerance"]),
        },
        "top_degree_node": degree_rows[0],
        "top_betweenness_node": node_betweenness[0],
        "top_flow_critical_node": node_ablation[0],
        "top_flow_critical_edge": edge_ablation[0],
        "worst_progressive_scenario": min(
            resilience, key=lambda row: row["resilience_index"]
        ),
        "worst_flood_scenario": worst_flood,
    }
    (output / "part3_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
