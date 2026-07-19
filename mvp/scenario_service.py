from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pyproj import Transformer

from mobility.scenario import ScenarioSpec
from mobility.simulation import SimulationEngine
from mvp.layers import RESULT_GEOJSON, LayerService


DOWNLOAD_ARTIFACTS = {
    "summary.json",
    "od_impacts.csv",
    "edge_rerouting.csv",
    *RESULT_GEOJSON,
}


class ScenarioService:
    def __init__(
        self,
        config: dict[str, Any],
        layers: LayerService,
    ) -> None:
        self.config = config
        self.paths = {key: Path(value) for key, value in config["paths"].items()}
        self.layers = layers
        simulation = config["simulation"]
        engine_config = {
            "paths": {
                "graph": str(self.paths["graph"]),
                "demand": str(self.paths["demand"]),
                "output": str(self.paths["sessions"]),
            },
            "assignment": {
                "bpr_alpha": simulation["bpr_alpha"],
                "bpr_beta": simulation["bpr_beta"],
                "max_iterations": simulation["max_iterations"],
                "tolerance": simulation["tolerance"],
            },
            "impact": {
                "affected_threshold": simulation["affected_threshold"],
                "grid_size_m": simulation["grid_size_m"],
                "route_examples": simulation["route_examples"],
            },
            "baseline_cache": str(self.paths["baseline"]),
        }
        self.engine = SimulationEngine(engine_config)
        self.to_graph_crs = Transformer.from_crs(
            config["map"]["target_crs"],
            config["map"]["source_crs"],
            always_xy=True,
        )
        self._baseline_signature = self.graph_signature()

    def graph_signature(self) -> tuple[int, int, float]:
        capacity = sum(
            float(data.get("relative_capacity", 0))
            for _, _, data in self.engine.graph.edges(data=True)
        )
        return (
            self.engine.graph.number_of_nodes(),
            self.engine.graph.number_of_edges(),
            round(capacity, 8),
        )

    def presets(self) -> list[dict[str, Any]]:
        rows = []
        for index in range(1, 10):
            scenario_id = f"D{index:03d}"
            directory = self.paths["phase4"] / "scenarios" / scenario_id
            scenario = json.loads(
                (directory / "scenario.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "name": scenario["name"],
                    "hazard_type": scenario["hazard_type"],
                    "summary": summary,
                }
            )
        return rows

    def preset(self, scenario_id: str) -> dict[str, Any] | None:
        if scenario_id not in {f"D{index:03d}" for index in range(1, 10)}:
            return None
        directory = self.paths["phase4"] / "scenarios" / scenario_id
        scenario = json.loads(
            (directory / "scenario.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        return {
            "result_id": scenario_id,
            "scenario": scenario,
            "summary": summary,
            "artifacts": self.artifact_urls(scenario_id, preset=True),
        }

    def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._run(request, "preview")

    def exact(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._run(request, "exact")

    def _run(
        self, request: dict[str, Any], mode: str
    ) -> dict[str, Any]:
        result_id = str(uuid.uuid4())
        actions = [self._translate_action(action) for action in request["actions"]]
        spec = ScenarioSpec.from_dict(
            {
                "scenario_id": result_id,
                "name": request["name"],
                "hazard_type": request["hazard_type"],
                "actions": actions,
            }
        )
        before = self.graph_signature()
        result = (
            self.engine.preview(spec)
            if mode == "preview"
            else self.engine.simulate(spec)
        )
        if before != self.graph_signature() or before != self._baseline_signature:
            raise RuntimeError("Baseline graph mutation detected")
        destination = self.paths["sessions"] / result_id
        result.to_directory(destination)
        return {
            "result_id": result_id,
            "summary": result.summary,
            "artifacts": self.artifact_urls(result_id, preset=False),
        }

    def _translate_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action = dict(action)
        if action["action"] == "close_circle":
            longitude = action.pop("longitude")
            latitude = action.pop("latitude")
            x, y = self.to_graph_crs.transform(longitude, latitude)
            action["x"] = x
            action["y"] = y
        return action

    def artifact_urls(
        self, result_id: str, preset: bool
    ) -> dict[str, str]:
        return {
            name: f"/api/v1/results/{result_id}/{name}"
            for name in sorted(DOWNLOAD_ARTIFACTS)
            if (
                self.paths["phase4"] / "scenarios" / result_id / name
                if preset
                else self.paths["sessions"] / result_id / name
            ).exists()
        }

    def artifact_path(
        self, result_id: str, artifact: str
    ) -> Path | None:
        if artifact not in DOWNLOAD_ARTIFACTS:
            return None
        preset = result_id in {f"D{index:03d}" for index in range(1, 10)}
        source = (
            self.paths["phase4"] / "scenarios" / result_id / artifact
            if preset
            else self.paths["sessions"] / result_id / artifact
        )
        if not source.exists():
            return None
        if artifact.endswith(".geojson"):
            if preset:
                return self.layers.preset_layer_path(result_id, artifact)
            target = source.parent / "web" / artifact
            if not target.exists() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
                self.layers.convert_geojson_file(source, target)
            return target
        return source
