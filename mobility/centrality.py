from __future__ import annotations

import random

import networkx as nx
import numpy as np


def _normalize_map(values: dict) -> dict:
    if not values:
        return {}
    minimum, maximum = min(values.values()), max(values.values())
    if maximum <= minimum:
        return {key: 0.0 for key in values}
    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in values.items()
    }


def degree_baseline(graph: nx.Graph) -> list[dict]:
    degree = {str(node): float(graph.degree(node)) for node in graph}
    capacity = {
        str(node): sum(
            float(data["relative_capacity"])
            for *_, data in graph.edges(node, data=True)
        )
        for node in graph
    }
    degree_norm = _normalize_map(degree)
    capacity_norm = _normalize_map(capacity)
    articulation = {str(node) for node in nx.articulation_points(graph)}
    rows = []
    for node in graph:
        key = str(node)
        rows.append(
            {
                "node_id": key,
                "x": float(graph.nodes[node]["x"]),
                "y": float(graph.nodes[node]["y"]),
                "degree": degree[key],
                "degree_centrality": degree[key] / max(len(graph) - 1, 1),
                "connected_capacity": capacity[key],
                "articulation_point": key in articulation,
                "degree_score": 0.70 * degree_norm[key]
                + 0.30 * capacity_norm[key],
            }
        )
    return sorted(rows, key=lambda row: row["degree_score"], reverse=True)


def betweenness_reference(
    graph: nx.Graph, source_count: int, seed: int
) -> tuple[list[dict], list[dict]]:
    count = min(source_count, len(graph))
    node_values = nx.betweenness_centrality(
        graph,
        k=count,
        normalized=True,
        weight="free_flow_time_min",
        seed=seed,
    )
    rng = random.Random(seed)
    sources = rng.sample(list(graph.nodes), count)
    edge_values = nx.edge_betweenness_centrality_subset(
        graph,
        sources=sources,
        targets=list(graph.nodes),
        normalized=True,
        weight="free_flow_time_min",
    )
    articulation = {str(node) for node in nx.articulation_points(graph)}
    bridges = {
        tuple(sorted((str(first), str(second))))
        for first, second in nx.bridges(graph)
    }
    node_rows = [
        {
            "node_id": str(node),
            "x": float(graph.nodes[node]["x"]),
            "y": float(graph.nodes[node]["y"]),
            "betweenness": float(value),
            "articulation_point": str(node) in articulation,
        }
        for node, value in node_values.items()
    ]
    edge_rows = [
        {
            "source": str(first),
            "target": str(second),
            "edge_betweenness": float(value),
            "graph_bridge": tuple(sorted((str(first), str(second)))) in bridges,
        }
        for (first, second), value in edge_values.items()
    ]
    return (
        sorted(node_rows, key=lambda row: row["betweenness"], reverse=True),
        sorted(edge_rows, key=lambda row: row["edge_betweenness"], reverse=True),
    )
