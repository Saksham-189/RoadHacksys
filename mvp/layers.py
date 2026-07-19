from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import networkx as nx
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry

from mobility.assignment import edge_key


CLIENT_LAYERS = (
    "baseline_network",
    "relative_flow",
    "node_criticality",
    "edge_criticality",
)

RESULT_GEOJSON = (
    "affected_zones.geojson",
    "node_impacts.geojson",
    "edge_rerouting.geojson",
    "route_examples.geojson",
    "disrupted_network.geojson",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class LayerService:
    def __init__(self, config: dict[str, Any], fingerprint: str) -> None:
        self.config = config
        self.paths = {key: Path(value) for key, value in config["paths"].items()}
        self.cache = self.paths["cache"] / "layers"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        source = config["map"]["source_crs"]
        target = config["map"]["target_crs"]
        self.transformer = Transformer.from_crs(
            source, target, always_xy=True
        )
        self.precision = int(config["map"]["coordinate_precision"])
        self.graph = nx.read_graphml(self.paths["graph"])
        self.bounds: list[list[float]] = []
        self.ensure_cache()

    def ensure_cache(self) -> None:
        manifest_path = self.cache / "manifest.json"
        expected = all(
            (self.cache / f"{name}.geojson").exists()
            for name in CLIENT_LAYERS
        )
        if manifest_path.exists() and expected:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == self.fingerprint:
                self.bounds = manifest["bounds"]
                self._ensure_preset_cache()
                return
        self._build_core_layers()
        self._ensure_preset_cache()

    def _coordinates(self, node_id: str) -> list[float]:
        data = self.graph.nodes[node_id]
        longitude, latitude = self.transformer.transform(
            float(data["x"]), float(data["y"])
        )
        return [
            round(longitude, self.precision),
            round(latitude, self.precision),
        ]

    def _build_core_layers(self) -> None:
        coordinates = {
            str(node): self._coordinates(str(node))
            for node in self.graph.nodes
        }
        longitudes = [value[0] for value in coordinates.values()]
        latitudes = [value[1] for value in coordinates.values()]
        self.bounds = [
            [min(latitudes), min(longitudes)],
            [max(latitudes), max(longitudes)],
        ]

        flow_rows = _csv_rows(self.paths["edge_flow"])
        flow = {
            edge_key(row["source"], row["target"]): row
            for row in flow_rows
        }
        node_rows = _csv_rows(self.paths["node_criticality"])
        nodes = {row["node_id"]: row for row in node_rows}
        edge_rows = _csv_rows(self.paths["edge_criticality"])
        edges = {
            edge_key(row["source"], row["target"]): row
            for row in edge_rows
        }

        baseline_features = []
        flow_features = []
        node_features = []
        edge_features = []
        for first, second, data in self.graph.edges(data=True):
            first_id, second_id = str(first), str(second)
            key = edge_key(first_id, second_id)
            geometry = {
                "type": "LineString",
                "coordinates": [coordinates[first_id], coordinates[second_id]],
            }
            base_properties = {
                "feature": "edge",
                "source": first_id,
                "target": second_id,
                "length_m": round(_number(data.get("length_m")), 2),
                "capacity": _number(data.get("relative_capacity"), 1),
                "speed_kmh": _number(data.get("speed_kmh")),
                "road_class": str(data.get("road_class", "unknown")),
            }
            baseline_features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": base_properties,
                }
            )
            flow_row = flow.get(key, {})
            flow_features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        **base_properties,
                        "flow": _number(flow_row.get("assigned_flow")),
                        "vc_ratio": _number(
                            flow_row.get("volume_capacity_ratio")
                        ),
                        "overloaded": str(
                            flow_row.get("overloaded", "False")
                        ).lower()
                        == "true",
                    },
                }
            )
            edge_row = edges.get(key, {})
            edge_features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        **base_properties,
                        "flow_criticality": _number(
                            edge_row.get("flow_criticality")
                        ),
                        "disconnected_demand_ratio": _number(
                            edge_row.get("disconnected_demand_ratio")
                        ),
                        "travel_time_increase": _number(
                            edge_row.get("travel_time_increase")
                        ),
                    },
                }
            )

        for node_id, data in self.graph.nodes(data=True):
            node_id = str(node_id)
            row = nodes.get(node_id, {})
            node_features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": coordinates[node_id],
                    },
                    "properties": {
                        "feature": "node",
                        "node_id": node_id,
                        "degree": _number(data.get("degree")),
                        "flow_criticality": _number(
                            row.get("flow_criticality")
                        ),
                        "disconnected_demand_ratio": _number(
                            row.get("disconnected_demand_ratio")
                        ),
                        "largest_component_loss": _number(
                            row.get("largest_component_loss")
                        ),
                    },
                }
            )

        self._write("baseline_network", baseline_features)
        self._write("relative_flow", flow_features)
        self._write("node_criticality", node_features)
        self._write("edge_criticality", edge_features)
        (self.cache / "manifest.json").write_text(
            json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "bounds": self.bounds,
                    "layers": list(CLIENT_LAYERS),
                    "source_crs": self.config["map"]["source_crs"],
                    "target_crs": self.config["map"]["target_crs"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write(self, name: str, features: list[dict]) -> None:
        (self.cache / f"{name}.geojson").write_text(
            json.dumps(
                {"type": "FeatureCollection", "features": features},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )

    def _ensure_preset_cache(self) -> None:
        destination = self.cache / "presets"
        destination.mkdir(parents=True, exist_ok=True)
        for index in range(1, 10):
            scenario_id = f"D{index:03d}"
            source_dir = self.paths["phase4"] / "scenarios" / scenario_id
            output_dir = destination / scenario_id
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in RESULT_GEOJSON:
                source = source_dir / name
                target = output_dir / name
                if not target.exists() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
                    self.convert_geojson_file(source, target)

    def convert_geojson_file(self, source: Path, target: Path) -> Path:
        payload = json.loads(source.read_text(encoding="utf-8"))
        features = []
        for feature in payload.get("features", []):
            geometry = feature.get("geometry")
            if geometry:
                projected = transform_geometry(
                    self.transformer.transform, shape(geometry)
                )
                geometry = self._round_geometry(mapping(projected))
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": feature.get("properties", {}),
                }
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {"type": "FeatureCollection", "features": features},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return target

    def _round_geometry(self, geometry: dict) -> dict:
        def round_values(value):
            if isinstance(value, (list, tuple)):
                return [round_values(item) for item in value]
            if isinstance(value, float):
                return round(value, self.precision)
            return value

        return {
            "type": geometry["type"],
            "coordinates": round_values(geometry["coordinates"]),
        }

    def layer_path(self, name: str) -> Path | None:
        if name not in CLIENT_LAYERS:
            return None
        return self.cache / f"{name}.geojson"

    def preset_layer_path(self, scenario_id: str, name: str) -> Path | None:
        if scenario_id not in {f"D{index:03d}" for index in range(1, 10)}:
            return None
        if name not in RESULT_GEOJSON:
            return None
        return self.cache / "presets" / scenario_id / name
