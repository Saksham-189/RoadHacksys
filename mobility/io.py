from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def graphml_safe_copy(graph: nx.Graph) -> nx.Graph:
    export = graph.__class__()
    export.graph.update(
        {
            key: value
            for key, value in graph.graph.items()
            if isinstance(value, (str, int, float, bool))
        }
    )
    for node, data in graph.nodes(data=True):
        export.add_node(
            node,
            **{
                key: value
                for key, value in data.items()
                if isinstance(value, (str, int, float, bool))
            },
        )
    if graph.is_multigraph():
        edges = graph.edges(keys=True, data=True)
        for source, target, key, data in edges:
            attributes = _safe_attributes(data)
            export.add_edge(source, target, key=key, **attributes)
    else:
        for source, target, data in graph.edges(data=True):
            export.add_edge(source, target, **_safe_attributes(data))
    return export


def _safe_attributes(data: dict) -> dict:
    attributes = {
        key: value
        for key, value in data.items()
        if isinstance(value, (str, int, float, bool))
    }
    for key, value in data.items():
        if isinstance(value, (list, tuple, dict)):
            attributes[f"{key}_json"] = json.dumps(value)
    return attributes


def write_graphml(graph: nx.Graph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graphml_safe_copy(graph), path)


def graph_to_geojson(
    graph: nx.Graph,
    path: str | Path,
    node_properties: dict[str, dict] | None = None,
    edge_properties: dict[tuple[str, str], dict] | None = None,
) -> None:
    node_properties = node_properties or {}
    edge_properties = edge_properties or {}
    features = []
    for node, data in graph.nodes(data=True):
        properties = {"feature": "node", "node_id": str(node), **node_properties.get(str(node), {})}
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(data["x"]), float(data["y"])],
                },
                "properties": properties,
            }
        )
    for source, target, data in graph.edges(data=True):
        source_data, target_data = graph.nodes[source], graph.nodes[target]
        properties = {
            "feature": "edge",
            "source": str(source),
            "target": str(target),
            **{
                key: value
                for key, value in data.items()
                if isinstance(value, (str, int, float, bool))
            },
            **edge_properties.get(_edge_key(source, target), {}),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(source_data["x"]), float(source_data["y"])],
                        [float(target_data["x"]), float(target_data["y"])],
                    ],
                },
                "properties": properties,
            }
        )
    Path(path).write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )


def _edge_key(first: object, second: object) -> tuple[str, str]:
    values = sorted((str(first), str(second)))
    return values[0], values[1]
