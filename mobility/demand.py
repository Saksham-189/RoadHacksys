from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .assignment import scale_demand_to_target
from .io import write_csv


def _normalize(values: np.ndarray) -> np.ndarray:
    minimum, maximum = float(values.min()), float(values.max())
    if maximum <= minimum:
        return np.ones_like(values)
    return (values - minimum) / (maximum - minimum)


def node_activity(graph: nx.Graph) -> dict[str, float]:
    nodes = list(graph.nodes)
    coordinates = np.asarray(
        [(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])) for node in nodes]
    )
    tree = cKDTree(coordinates)
    edge_lengths = {
        str(node): sum(float(data["length_m"]) for *_, data in graph.edges(node, data=True))
        for node in nodes
    }
    local_density = []
    for index, node in enumerate(nodes):
        nearby = tree.query_ball_point(coordinates[index], 500.0)
        local_density.append(
            sum(edge_lengths[str(nodes[position])] for position in nearby)
        )
    degrees = np.asarray([graph.degree(node) for node in nodes], dtype=float)
    capacities = np.asarray(
        [
            sum(float(data["relative_capacity"]) for *_, data in graph.edges(node, data=True))
            for node in nodes
        ],
        dtype=float,
    )
    activity = (
        0.50 * _normalize(np.asarray(local_density))
        + 0.30 * _normalize(degrees)
        + 0.20 * _normalize(capacities)
    )
    activity = np.maximum(activity, 0.01)
    return {str(node): float(value) for node, value in zip(nodes, activity)}


def generate_demands(
    graph: nx.Graph, config: dict, output: Path
) -> tuple[list[dict], list[dict], dict]:
    settings = config["demand"]
    rng = np.random.default_rng(int(config["seed"]))
    activities = node_activity(graph)
    nodes = np.asarray(list(activities), dtype=object)
    weights = np.asarray([activities[node] for node in nodes], dtype=float)
    weights /= weights.sum()
    origin_count = min(int(settings["origins"]), len(nodes))
    origins = rng.choice(nodes, size=origin_count, replace=False, p=weights)
    pairs = []
    for origin in origins:
        candidates = nodes[nodes != origin]
        candidate_weights = np.asarray(
            [activities[node] for node in candidates], dtype=float
        )
        candidate_weights /= candidate_weights.sum()
        count = min(int(settings["destinations_per_origin"]), len(candidates))
        destinations = rng.choice(
            candidates, size=count, replace=False, p=candidate_weights
        )
        origin_data = graph.nodes[str(origin)]
        for destination in destinations:
            destination_data = graph.nodes[str(destination)]
            distance = float(
                np.hypot(
                    float(origin_data["x"]) - float(destination_data["x"]),
                    float(origin_data["y"]) - float(destination_data["y"]),
                )
            )
            gravity = (
                activities[str(origin)] * activities[str(destination)]
                / (
                    distance + float(settings["distance_offset_m"])
                )
                ** float(settings["distance_exponent"])
            )
            pairs.append(
                {
                    "origin": str(origin),
                    "destination": str(destination),
                    "euclidean_distance_m": distance,
                    "demand": gravity,
                }
            )
    gravity, gravity_factor = scale_demand_to_target(
        graph, pairs, float(settings["target_p90_vc"])
    )
    uniform_raw = [{**row, "demand": 1.0} for row in pairs]
    uniform, uniform_factor = scale_demand_to_target(
        graph, uniform_raw, float(settings["target_p90_vc"])
    )
    write_csv(output / "gravity_demand.csv", gravity)
    write_csv(output / "uniform_demand.csv", uniform)
    summary = {
        "origins": len(set(row["origin"] for row in pairs)),
        "od_pairs": len(pairs),
        "gravity_scale_factor": gravity_factor,
        "uniform_scale_factor": uniform_factor,
        "target_p90_vc": settings["target_p90_vc"],
        "statement": "Relative graph-derived demand; not measured traffic.",
    }
    (output / "demand_summary.json").write_text(json.dumps(summary, indent=2))
    return gravity, uniform, summary
