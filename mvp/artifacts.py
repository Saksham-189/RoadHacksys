from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


REQUIRED_BASELINE = (
    "baseline_summary.json",
    "baseline_od_routes.csv",
    "baseline_routes.json.gz",
    "baseline_edge_state.csv",
)

REQUIRED_SCENARIO = (
    "summary.json",
    "scenario.json",
    "affected_zones.geojson",
    "edge_rerouting.geojson",
    "route_examples.geojson",
    "disrupted_network.geojson",
)

REQUIRED_STATIC = (
    "index.html",
    "app.css",
    "app.js",
    "vendor/leaflet/leaflet.css",
    "vendor/leaflet/leaflet.js",
    "vendor/lucide/lucide.min.js",
)


def file_signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


@dataclass
class ArtifactStatus:
    status: str
    missing: list[str]
    warnings: list[str]
    fingerprint: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "missing": self.missing,
            "warnings": self.warnings,
            "fingerprint": self.fingerprint,
            "details": self.details,
        }


class ArtifactRegistry:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.paths = {key: Path(value) for key, value in config["paths"].items()}
        self.cache = self.paths["cache"]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sessions = self.paths["sessions"]
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.inference = self.paths["inference"]
        self.inference.mkdir(parents=True, exist_ok=True)
        self.status = self.validate()

    def validate(self) -> ArtifactStatus:
        missing: list[str] = []
        warnings: list[str] = []
        files = ("checkpoint", "manifest", "graph", "demand")
        for key in files:
            if not self.paths[key].is_file():
                missing.append(str(self.paths[key]))
        for name in REQUIRED_BASELINE:
            path = self.paths["baseline"] / name
            if not path.is_file():
                missing.append(str(path))
        for scenario_id in (f"D{index:03d}" for index in range(1, 10)):
            for name in REQUIRED_SCENARIO:
                path = self.paths["phase4"] / "scenarios" / scenario_id / name
                if not path.is_file():
                    missing.append(str(path))
        for name in REQUIRED_STATIC:
            path = self.paths["static"] / name
            if not path.is_file():
                warnings.append(f"Static asset unavailable: {path}")

        details: dict[str, Any] = {}
        fingerprint = None
        if not any(
            str(self.paths[key]) in item
            for key in ("graph", "demand")
            for item in missing
        ):
            signatures = {
                key: file_signature(self.paths[key])
                for key in ("graph", "demand")
            }
            fingerprint = hashlib.sha256(
                json.dumps(signatures, sort_keys=True).encode("utf-8")
            ).hexdigest()
            details["source_signatures"] = signatures
            self._validate_or_create_baseline_fingerprint(
                fingerprint, signatures, warnings
            )
            graph = nx.read_graphml(self.paths["graph"])
            with self.paths["demand"].open(
                newline="", encoding="utf-8"
            ) as handle:
                demand_rows = sum(1 for _ in csv.DictReader(handle))
            details.update(
                {
                    "graph_nodes": graph.number_of_nodes(),
                    "graph_edges": graph.number_of_edges(),
                    "demand_pairs": demand_rows,
                }
            )
        status = "missing" if missing else ("degraded" if warnings else "ready")
        return ArtifactStatus(status, missing, warnings, fingerprint, details)

    def _validate_or_create_baseline_fingerprint(
        self,
        fingerprint: str,
        signatures: dict[str, Any],
        warnings: list[str],
    ) -> None:
        path = self.cache / "baseline_fingerprint.json"
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("fingerprint") != fingerprint:
                warnings.append(
                    "Baseline cache fingerprint is stale; rebuild Phase 4 baseline"
                )
            return
        summary_path = self.paths["baseline"] / "baseline_summary.json"
        if not summary_path.exists():
            return
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        graph = nx.read_graphml(self.paths["graph"])
        if (
            int(summary.get("nodes", -1)) == graph.number_of_nodes()
            and int(summary.get("edges", -1)) == graph.number_of_edges()
        ):
            path.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "sources": signatures,
                        "baseline_summary": str(summary_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            warnings.append("Baseline summary does not match transport graph")

    def preset_ids(self) -> list[str]:
        return [f"D{index:03d}" for index in range(1, 10)]

