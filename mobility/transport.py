from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import rasterio
from scipy import ndimage

from .io import graph_to_geojson, write_graphml


def _pixels(data: dict) -> list[list[int]]:
    raw = data.get("pixels_json", data.get("pixels", "[]"))
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return list(raw)


def _float_node_attributes(graph: nx.Graph) -> None:
    for _, data in graph.nodes(data=True):
        for name in ("row", "col", "x", "y"):
            data[name] = float(data[name])


def _classify_width(
    width: float, lower: float, upper: float
) -> str:
    if width <= lower:
        return "narrow"
    if width <= upper:
        return "medium"
    return "wide"


def prepare_transport_graph(config: dict) -> tuple[nx.Graph, dict]:
    paths = config["paths"]
    source_path = (
        Path(paths["consolidation_output"]) / "consolidated_graph.graphml"
    )
    mask_path = (
        Path(paths["consolidation_output"]) / "consolidated_mask.tif"
    )
    output = Path(paths["part3_output"])
    output.mkdir(parents=True, exist_ok=True)

    source = nx.read_graphml(source_path)
    _float_node_attributes(source)
    source_simple = nx.Graph(source)
    source_simple.remove_edges_from(nx.selfloop_edges(source_simple))
    components = list(nx.connected_components(source_simple))
    largest_nodes = max(components, key=len)
    core_multi = source.subgraph(largest_nodes).copy()

    with rasterio.open(mask_path) as raster:
        mask = raster.read(1).astype(bool)
        resolution_m = float(abs(raster.transform.a))
    distance = ndimage.distance_transform_edt(mask)

    edge_widths = []
    for _, _, _, data in core_multi.edges(keys=True, data=True):
        samples = []
        for row, column in _pixels(data):
            row, column = int(row), int(column)
            if 0 <= row < distance.shape[0] and 0 <= column < distance.shape[1]:
                samples.append(distance[row, column])
        width_proxy_m = (
            2.0 * float(np.median(samples)) * resolution_m
            if samples
            else 2.0 * resolution_m
        )
        data["width_proxy_m"] = width_proxy_m
        data["estimated_width_m"] = float(np.clip(width_proxy_m, 5.0, 30.0))
        edge_widths.append(width_proxy_m)

    quantiles = config["capacity"]["width_quantiles"]
    lower, upper = np.quantile(edge_widths, quantiles)
    classes = config["capacity"]["classes"]
    penalty_weight = float(config["capacity"]["healing_penalty_weight"])
    for _, _, _, data in core_multi.edges(keys=True, data=True):
        road_class = _classify_width(
            float(data["width_proxy_m"]), float(lower), float(upper)
        )
        capacity = float(classes[road_class]["relative_capacity"])
        speed = float(classes[road_class]["speed_kmh"])
        length = float(data["length_m"])
        confidence = float(data.get("healing_confidence", 1.0))
        penalty = 1.0 + penalty_weight * (1.0 - confidence)
        free_time = length / (speed * 1000.0 / 60.0) * penalty
        data.update(
            {
                "road_class": road_class,
                "relative_capacity": capacity,
                "speed_kmh": speed,
                "healing_penalty": penalty,
                "free_flow_time_min": free_time,
                "travel_time_min": free_time,
            }
        )

    transport = nx.Graph()
    transport.graph.update(
        {
            "crs": source.graph.get("crs", "EPSG:32643"),
            "capacity_units": "relative",
            "traffic_units": "relative_demand",
            "coverage_gate_met": source.graph.get("coverage_gate_met", False),
        }
    )
    transport.add_nodes_from(core_multi.nodes(data=True))
    removed_invalid = 0
    merged_parallel = 0
    for first, second, _, data in core_multi.edges(keys=True, data=True):
        if first == second or float(data.get("length_m", 0)) <= 0:
            removed_invalid += 1
            continue
        if transport.has_edge(first, second):
            existing = transport[first][second]
            existing["relative_capacity"] += float(data["relative_capacity"])
            existing["width_proxy_m"] = max(
                float(existing["width_proxy_m"]), float(data["width_proxy_m"])
            )
            existing["estimated_width_m"] = max(
                float(existing["estimated_width_m"]), float(data["estimated_width_m"])
            )
            if float(data["free_flow_time_min"]) < float(
                existing["free_flow_time_min"]
            ):
                capacity = existing["relative_capacity"]
                existing.update(data)
                existing["relative_capacity"] = capacity
            existing["parallel_count"] = int(existing.get("parallel_count", 1)) + 1
            merged_parallel += 1
        else:
            attributes = dict(data)
            attributes["parallel_count"] = 1
            transport.add_edge(first, second, **attributes)

    graph_path = output / "transport_graph.graphml"
    write_graphml(transport, graph_path)
    graph_to_geojson(transport, output / "transport_edges.geojson")
    report = {
        "source_graph": str(source_path),
        "source_nodes": source.number_of_nodes(),
        "source_edges": source.number_of_edges(),
        "source_components": len(components),
        "routing_nodes": transport.number_of_nodes(),
        "routing_edges": transport.number_of_edges(),
        "routing_node_coverage": transport.number_of_nodes()
        / max(source.number_of_nodes(), 1),
        "removed_invalid_edges": removed_invalid,
        "merged_parallel_edges": merged_parallel,
        "width_proxy_quantiles_m": {
            "lower": float(lower),
            "upper": float(upper),
        },
        "width_warning": (
            "Proxy values classify relative corridor width; physical estimates "
            "are capped at 30 m because Sentinel-2 cannot resolve lanes."
        ),
        "capacity_statement": (
            "Relative satellite-derived capacity; not calibrated vehicle counts."
        ),
        "output_graph": str(graph_path),
    }
    (output / "graph_preparation_report.json").write_text(
        json.dumps(report, indent=2)
    )
    return transport, report
