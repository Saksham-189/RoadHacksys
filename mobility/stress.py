from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

from .assignment import AssignmentResult, assign_msa, edge_key


def assignment_parameters(config: dict, stress: bool = False) -> dict:
    settings = config["assignment"]
    return {
        "alpha": float(settings["bpr_alpha"]),
        "beta": float(settings["bpr_beta"]),
        "max_iterations": int(
            settings["stress_max_iterations"]
            if stress
            else settings["max_iterations"]
        ),
        "tolerance": float(settings["tolerance"]),
    }


def largest_component_ratio(graph: nx.Graph, baseline_nodes: int) -> float:
    if not graph:
        return 0.0
    largest = len(max(nx.connected_components(graph), key=len))
    return largest / max(baseline_nodes, 1)


def consequence_metrics(
    baseline: AssignmentResult,
    perturbed: AssignmentResult,
    graph: nx.Graph,
    baseline_nodes: int,
) -> dict[str, float]:
    served_ratio = float(np.clip(perturbed.served_demand_ratio, 0.0, 1.0))
    if baseline.mean_travel_time > 0 and perturbed.mean_travel_time > 0:
        travel_increase = max(
            0.0,
            perturbed.mean_travel_time / baseline.mean_travel_time - 1.0,
        )
        time_ratio = min(
            1.0, baseline.mean_travel_time / perturbed.mean_travel_time
        )
    else:
        travel_increase = 1.0
        time_ratio = 0.0
    efficiency_loss = max(
        0.0,
        1.0
        - perturbed.global_efficiency
        / max(baseline.global_efficiency, 1e-12),
    )
    largest_ratio = largest_component_ratio(graph, baseline_nodes)
    return {
        "served_demand_ratio": served_ratio,
        "disconnected_demand_ratio": float(
            np.clip(1.0 - served_ratio, 0.0, 1.0)
        ),
        "mean_travel_time_min": perturbed.mean_travel_time,
        "travel_time_increase": travel_increase,
        "global_efficiency": perturbed.global_efficiency,
        "efficiency_loss": efficiency_loss,
        "largest_component_ratio": largest_ratio,
        "largest_component_loss": 1.0 - largest_ratio,
        "overloaded_edges": float(perturbed.overloaded_edges),
        "overload_increase": float(
            max(0, perturbed.overloaded_edges - baseline.overloaded_edges)
        ),
        "resilience_index": float(
            np.clip(served_ratio * time_ratio, 0.0, 1.0)
        ),
        "assignment_iterations": float(perturbed.iterations),
        "assignment_convergence": perturbed.convergence,
    }


def node_throughput(
    graph: nx.Graph, flows: dict[tuple[str, str], float]
) -> dict[str, float]:
    throughput = {str(node): 0.0 for node in graph}
    for first, second in graph.edges:
        flow = flows.get(edge_key(first, second), 0.0)
        throughput[str(first)] += flow
        throughput[str(second)] += flow
    return throughput


def _rank_ids(rows: list[dict], id_field: str, score_field: str) -> list[str]:
    return [
        str(row[id_field])
        for row in sorted(
            rows, key=lambda row: float(row[score_field]), reverse=True
        )
    ]


def build_ablation_candidates(
    graph: nx.Graph,
    baseline: AssignmentResult,
    degree_rows: list[dict],
    node_betweenness_rows: list[dict],
    edge_betweenness_rows: list[dict],
    count: int,
) -> tuple[list[str], list[tuple[str, str]], dict[str, float]]:
    throughput = node_throughput(graph, baseline.edge_flows)
    throughput_rank = sorted(throughput, key=throughput.get, reverse=True)
    node_candidates = []
    for sequence in (
        throughput_rank[:count],
        _rank_ids(degree_rows, "node_id", "degree_score")[:count],
        _rank_ids(node_betweenness_rows, "node_id", "betweenness")[:count],
    ):
        for node in sequence:
            if node not in node_candidates:
                node_candidates.append(node)

    flow_rank = sorted(
        baseline.edge_flows, key=baseline.edge_flows.get, reverse=True
    )
    betweenness_rank = [
        edge_key(row["source"], row["target"])
        for row in sorted(
            edge_betweenness_rows,
            key=lambda row: float(row["edge_betweenness"]),
            reverse=True,
        )[:count]
    ]
    edge_candidates = []
    for sequence in (flow_rank[:count], betweenness_rank):
        for edge in sequence:
            edge = edge_key(*edge)
            if edge not in edge_candidates:
                edge_candidates.append(edge)
    return node_candidates, edge_candidates, throughput


def run_ablations(
    graph: nx.Graph,
    demand: list[dict],
    baseline: AssignmentResult,
    node_candidates: list[str],
    edge_candidates: list[tuple[str, str]],
    config: dict,
) -> tuple[list[dict], list[dict]]:
    parameters = assignment_parameters(config, stress=True)
    baseline_nodes = graph.number_of_nodes()
    node_rows = []
    for index, node in enumerate(node_candidates, start=1):
        perturbed_graph = graph.copy()
        if node in perturbed_graph:
            perturbed_graph.remove_node(node)
        result = assign_msa(perturbed_graph, demand, **parameters)
        node_rows.append(
            {
                "node_id": node,
                **consequence_metrics(
                    baseline, result, perturbed_graph, baseline_nodes
                ),
            }
        )
        print(
            f"node ablation [{index}/{len(node_candidates)}] {node}",
            flush=True,
        )

    edge_rows = []
    for index, edge in enumerate(edge_candidates, start=1):
        perturbed_graph = graph.copy()
        if perturbed_graph.has_edge(*edge):
            perturbed_graph.remove_edge(*edge)
        result = assign_msa(perturbed_graph, demand, **parameters)
        edge_rows.append(
            {
                "source": edge[0],
                "target": edge[1],
                **consequence_metrics(
                    baseline, result, perturbed_graph, baseline_nodes
                ),
            }
        )
        print(
            f"edge ablation [{index}/{len(edge_candidates)}] {edge}",
            flush=True,
        )

    _add_criticality_scores(node_rows)
    _add_criticality_scores(edge_rows)
    return (
        sorted(node_rows, key=lambda row: row["flow_criticality"], reverse=True),
        sorted(edge_rows, key=lambda row: row["flow_criticality"], reverse=True),
    )


def _add_criticality_scores(rows: list[dict]) -> None:
    fields = (
        "travel_time_increase",
        "disconnected_demand_ratio",
        "overload_increase",
        "efficiency_loss",
        "largest_component_loss",
    )
    normalized = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows])
        minimum, maximum = float(values.min()), float(values.max())
        normalized[field] = (
            (values - minimum) / (maximum - minimum)
            if maximum > minimum
            else np.zeros_like(values)
        )
    for index, row in enumerate(rows):
        row["flow_criticality"] = float(
            0.30 * normalized["travel_time_increase"][index]
            + 0.25 * normalized["disconnected_demand_ratio"][index]
            + 0.20 * normalized["overload_increase"][index]
            + 0.15 * normalized["efficiency_loss"][index]
            + 0.10 * normalized["largest_component_loss"][index]
        )


def run_progressive_scenarios(
    graph: nx.Graph,
    demand: list[dict],
    baseline: AssignmentResult,
    degree_rows: list[dict],
    betweenness_rows: list[dict],
    node_ablation_rows: list[dict],
    throughput: dict[str, float],
    config: dict,
) -> list[dict]:
    settings = config["stress"]
    parameters = assignment_parameters(config, stress=True)
    baseline_nodes = graph.number_of_nodes()
    all_nodes = list(graph.nodes)
    rng = random.Random(int(config["seed"]))
    random_rank = all_nodes.copy()
    rng.shuffle(random_rank)
    degree_rank = _rank_ids(degree_rows, "node_id", "degree_score")
    betweenness_rank = _rank_ids(
        betweenness_rows, "node_id", "betweenness"
    )
    critical_rank = _rank_ids(
        node_ablation_rows, "node_id", "flow_criticality"
    )
    throughput_rank = sorted(throughput, key=throughput.get, reverse=True)
    critical_rank.extend(
        node for node in throughput_rank if node not in critical_rank
    )
    rankings = {
        "random": random_rank,
        "highest_degree": degree_rank,
        "highest_betweenness": betweenness_rank,
        "highest_flow_criticality": critical_rank,
    }
    rows = []
    for strategy, ranking in rankings.items():
        for percentage in settings["removal_percentages"]:
            count = max(1, math.ceil(float(percentage) * baseline_nodes))
            removed = ranking[:count]
            perturbed = graph.copy()
            perturbed.remove_nodes_from(removed)
            result = assign_msa(perturbed, demand, **parameters)
            rows.append(
                {
                    "strategy": strategy,
                    "removal_percentage": float(percentage),
                    "removed_nodes": len(removed),
                    **consequence_metrics(
                        baseline, result, perturbed, baseline_nodes
                    ),
                }
            )
            print(
                f"scenario {strategy} {float(percentage):.0%}", flush=True
            )
    return rows


def run_flood_scenarios(
    graph: nx.Graph,
    demand: list[dict],
    baseline: AssignmentResult,
    node_ablation_rows: list[dict],
    config: dict,
) -> list[dict]:
    settings = config["stress"]
    parameters = assignment_parameters(config, stress=True)
    baseline_nodes = graph.number_of_nodes()
    centers = node_ablation_rows[: int(settings["flood_centers"])]
    rows = []
    for center_rank, center in enumerate(centers, start=1):
        node = center["node_id"]
        x, y = float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])
        for radius in settings["flood_radii_m"]:
            removed = [
                candidate
                for candidate, data in graph.nodes(data=True)
                if math.hypot(float(data["x"]) - x, float(data["y"]) - y)
                <= float(radius)
            ]
            perturbed = graph.copy()
            perturbed.remove_nodes_from(removed)
            result, cascade_rounds, degraded = _run_cascade(
                perturbed, demand, config
            )
            rows.append(
                {
                    "center_rank": center_rank,
                    "center_node": node,
                    "center_x": x,
                    "center_y": y,
                    "radius_m": float(radius),
                    "removed_nodes": len(removed),
                    "cascade_rounds": cascade_rounds,
                    "capacity_degraded_edges": degraded,
                    **consequence_metrics(
                        baseline, result, perturbed, baseline_nodes
                    ),
                }
            )
            print(
                f"flood center={center_rank} radius={radius}m", flush=True
            )
    return rows


def _run_cascade(
    graph: nx.Graph, demand: list[dict], config: dict
) -> tuple[AssignmentResult, int, int]:
    cascade = config["stress"]["cascade"]
    parameters = assignment_parameters(config, stress=True)
    result = assign_msa(graph, demand, **parameters)
    if not cascade["enabled"]:
        return result, 0, 0
    degraded_edges: set[tuple[str, str]] = set()
    rounds = 0
    for rounds in range(1, int(cascade["max_rounds"]) + 1):
        newly_degraded = []
        for first, second, data in graph.edges(data=True):
            key = edge_key(first, second)
            capacity = float(data["relative_capacity"])
            flow = result.edge_flows.get(key, 0.0)
            if (
                flow / max(capacity, 1e-12)
                > float(cascade["overload_threshold"])
                and key not in degraded_edges
            ):
                newly_degraded.append((first, second, key))
        if not newly_degraded:
            rounds -= 1
            break
        for first, second, key in newly_degraded:
            graph[first][second]["relative_capacity"] *= float(
                cascade["capacity_factor"]
            )
            degraded_edges.add(key)
        result = assign_msa(graph, demand, **parameters)
    return result, rounds, len(degraded_edges)
