from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
from pyproj import Transformer
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform as transform_geometry

from .assignment import edge_key


SUPPORTED_ACTIONS = {
    "close_nodes",
    "close_edges",
    "capacity_derating",
    "close_circle",
    "close_polygon",
}


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    name: str
    hazard_type: str
    actions: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenarioSpec":
        required = ("scenario_id", "name", "hazard_type", "actions")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Scenario missing keys: {', '.join(missing)}")
        if not data["scenario_id"] or not isinstance(data["actions"], list):
            raise ValueError("Scenario ID must be non-empty and actions must be a list")
        for action in data["actions"]:
            if action.get("action") not in SUPPORTED_ACTIONS:
                raise ValueError(f"Unsupported action: {action.get('action')}")
        return cls(
            str(data["scenario_id"]),
            str(data["name"]),
            str(data["hazard_type"]),
            deepcopy(data["actions"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ScenarioSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "hazard_type": self.hazard_type,
            "actions": deepcopy(self.actions),
        }


@dataclass
class ResolvedScenario:
    graph: nx.Graph
    closed_nodes: list[dict]
    closed_edges: list[dict]
    degraded_edges: list[dict]
    spec: ScenarioSpec

    def metadata(self) -> dict:
        return {
            **self.spec.to_dict(),
            "closed_node_count": len(self.closed_nodes),
            "closed_edge_count": len(self.closed_edges),
            "degraded_edge_count": len(self.degraded_edges),
        }


def resolve_scenario(graph: nx.Graph, spec: ScenarioSpec) -> ResolvedScenario:
    working = graph.copy()
    closed_nodes: dict[str, dict] = {}
    closed_edges: dict[tuple[str, str], dict] = {}
    degraded_edges: dict[tuple[str, str], dict] = {}

    for action in spec.actions:
        kind = action["action"]
        if kind == "close_nodes":
            node_ids = [str(node) for node in action.get("node_ids", [])]
            unknown = [node for node in node_ids if node not in working]
            if unknown:
                raise ValueError(f"Unknown node IDs: {unknown}")
            for node in node_ids:
                for neighbor in list(working.neighbors(node)):
                    key = edge_key(node, neighbor)
                    closed_edges.setdefault(
                        key, {"source": key[0], "target": key[1], "reason": kind}
                    )
                data = working.nodes[node]
                closed_nodes[node] = {
                    "node_id": node,
                    "x": float(data["x"]),
                    "y": float(data["y"]),
                    "reason": kind,
                }
            working.remove_nodes_from(node_ids)
        elif kind in {"close_edges", "capacity_derating"}:
            entries = action.get("edges", [])
            for entry in entries:
                first, second = str(entry["source"]), str(entry["target"])
                if not working.has_edge(first, second):
                    raise ValueError(f"Unknown edge: {first}-{second}")
                key = edge_key(first, second)
                if kind == "close_edges":
                    closed_edges[key] = {
                        "source": key[0],
                        "target": key[1],
                        "reason": kind,
                    }
                    working.remove_edge(first, second)
                else:
                    factor = float(entry.get("capacity_factor", action.get("capacity_factor", 0)))
                    if not 0 < factor <= 1:
                        raise ValueError("Capacity factor must be in (0, 1]")
                    before = float(working[first][second]["relative_capacity"])
                    working[first][second]["relative_capacity"] = before * factor
                    degraded_edges[key] = {
                        "source": key[0],
                        "target": key[1],
                        "capacity_before": before,
                        "capacity_after": before * factor,
                        "capacity_factor": factor,
                        "reason": kind,
                    }
        elif kind == "close_circle":
            radius = float(action.get("radius_m", 0))
            if radius <= 0:
                raise ValueError("Circle radius must be positive")
            region = Point(float(action["x"]), float(action["y"])).buffer(radius)
            _close_region(
                working, region, kind, closed_nodes, closed_edges
            )
        elif kind == "close_polygon":
            geometry_data = action.get("geometry")
            if geometry_data is None and action.get("geometry_path"):
                geometry_data = json.loads(Path(action["geometry_path"]).read_text())
            if not geometry_data:
                raise ValueError("Polygon action requires geometry or geometry_path")
            if geometry_data.get("type") == "Feature":
                geometry_data = geometry_data["geometry"]
            region = shape(geometry_data)
            if not region.is_valid or region.is_empty:
                raise ValueError("Polygon geometry is invalid or empty")
            source_crs = action.get("crs", graph.graph.get("crs", "EPSG:32643"))
            target_crs = graph.graph.get("crs", "EPSG:32643")
            if str(source_crs) != str(target_crs):
                projector = Transformer.from_crs(
                    source_crs, target_crs, always_xy=True
                )
                region = transform_geometry(projector.transform, region)
            _close_region(
                working, region, kind, closed_nodes, closed_edges
            )
    return ResolvedScenario(
        working,
        list(closed_nodes.values()),
        list(closed_edges.values()),
        list(degraded_edges.values()),
        spec,
    )


def _close_region(
    graph: nx.Graph,
    region,
    reason: str,
    closed_nodes: dict[str, dict],
    closed_edges: dict[tuple[str, str], dict],
) -> None:
    nodes = [
        str(node)
        for node, data in graph.nodes(data=True)
        if region.covers(Point(float(data["x"]), float(data["y"])))
    ]
    node_set = set(nodes)
    edges = []
    for first, second in list(graph.edges):
        first_data, second_data = graph.nodes[first], graph.nodes[second]
        segment = LineString(
            [
                (float(first_data["x"]), float(first_data["y"])),
                (float(second_data["x"]), float(second_data["y"])),
            ]
        )
        if (
            str(first) in node_set
            or str(second) in node_set
            or region.intersects(segment)
        ):
            edges.append((str(first), str(second)))
    for node in nodes:
        data = graph.nodes[node]
        closed_nodes[node] = {
            "node_id": node,
            "x": float(data["x"]),
            "y": float(data["y"]),
            "reason": reason,
        }
    for first, second in edges:
        key = edge_key(first, second)
        closed_edges[key] = {
            "source": key[0],
            "target": key[1],
            "reason": reason,
        }
    graph.remove_nodes_from(nodes)
    graph.remove_edges_from(
        [(first, second) for first, second in edges if graph.has_edge(first, second)]
    )
