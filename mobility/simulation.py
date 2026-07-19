from __future__ import annotations

import csv
import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra

from .assignment import (
    AssignmentResult,
    SparseAssignmentNetwork,
    assign_msa,
    bpr_time,
    edge_key,
)
from .config import load_config
from .io import graph_to_geojson, write_csv
from .scenario import ResolvedScenario, ScenarioSpec, resolve_scenario


def _read_demand(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            {
                "origin": row["origin"],
                "destination": row["destination"],
                "demand": float(row["demand"]),
            }
            for row in csv.DictReader(handle)
        ]


def _route_records(
    graph: nx.Graph,
    demand: list[dict],
    edge_costs: dict[tuple[str, str], float],
) -> list[dict]:
    network = SparseAssignmentNetwork(graph)
    costs = np.asarray(
        [
            edge_costs.get(key, network.free_flow_time[index])
            for index, key in enumerate(network.keys)
        ],
        dtype=float,
    )
    valid_origins = sorted(
        {
            str(row["origin"])
            for row in demand
            if str(row["origin"]) in network.node_index
        }
    )
    if valid_origins:
        origin_indices = [network.node_index[node] for node in valid_origins]
        distances, predecessors = dijkstra(
            network.matrix(costs),
            directed=False,
            indices=origin_indices,
            return_predecessors=True,
        )
        origin_rows = {
            origin: index for index, origin in enumerate(valid_origins)
        }
    else:
        distances = predecessors = None
        origin_rows = {}

    records = []
    for row in demand:
        origin, destination = str(row["origin"]), str(row["destination"])
        amount = float(row["demand"])
        record = {
            "origin": origin,
            "destination": destination,
            "demand": amount,
            "served": False,
            "travel_time_min": None,
            "distance_m": None,
            "route_nodes": [],
            "route_edges": [],
        }
        if (
            origin not in network.node_index
            or destination not in network.node_index
            or origin not in origin_rows
        ):
            records.append(record)
            continue
        source = network.node_index[origin]
        target = network.node_index[destination]
        source_row = origin_rows[origin]
        travel_time = float(distances[source_row, target])
        if not np.isfinite(travel_time):
            records.append(record)
            continue
        current = target
        route_nodes = [str(network.nodes[current])]
        route_edges = []
        distance_m = 0.0
        valid = True
        while current != source:
            previous = int(predecessors[source_row, current])
            if previous < 0:
                valid = False
                break
            first, second = str(network.nodes[previous]), str(network.nodes[current])
            key = edge_key(first, second)
            route_edges.append(key)
            distance_m += float(graph[first][second]["length_m"])
            route_nodes.append(first)
            current = previous
        if valid:
            record.update(
                {
                    "served": True,
                    "travel_time_min": travel_time,
                    "distance_m": distance_m,
                    "route_nodes": list(reversed(route_nodes)),
                    "route_edges": list(reversed(route_edges)),
                }
            )
        records.append(record)
    return records


def _assignment_from_preview(
    graph: nx.Graph,
    demand: list[dict],
    baseline_flows: dict[tuple[str, str], float],
    alpha: float,
    beta: float,
) -> AssignmentResult:
    network = SparseAssignmentNetwork(graph)
    flows = np.asarray(
        [baseline_flows.get(key, 0.0) for key in network.keys], dtype=float
    )
    costs = bpr_time(
        network.free_flow_time, flows, network.capacity, alpha, beta
    )
    assigned, metrics = network.all_or_nothing(demand, costs)
    final_costs = bpr_time(
        network.free_flow_time, assigned, network.capacity, alpha, beta
    )
    ratios = np.divide(
        assigned,
        network.capacity,
        out=np.zeros_like(assigned),
        where=network.capacity > 0,
    )
    return AssignmentResult(
        {key: float(value) for key, value in zip(network.keys, assigned)},
        {key: float(value) for key, value in zip(network.keys, final_costs)},
        float(metrics["served_demand"]),
        float(metrics["total_demand"]),
        float(metrics["mean_travel_time"]),
        float(metrics["global_efficiency"]),
        int((ratios > 1).sum()),
        1,
        0.0,
    )


@dataclass
class SimulationResult:
    scenario: ScenarioSpec
    mode: str
    resolved: ResolvedScenario
    assignment: AssignmentResult
    summary: dict[str, Any]
    od_impacts: list[dict]
    edge_rerouting: list[dict]
    affected_zones: dict
    node_impacts: dict
    route_examples: dict
    runtime_seconds: float
    baseline_graph: nx.Graph

    def to_directory(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        (output / "scenario.json").write_text(
            json.dumps(self.scenario.to_dict(), indent=2)
        )
        (output / "resolved_scenario.json").write_text(
            json.dumps(self.resolved.metadata(), indent=2)
        )
        write_csv(output / "closed_nodes.csv", self.resolved.closed_nodes)
        write_csv(output / "closed_edges.csv", self.resolved.closed_edges)
        write_csv(output / "degraded_edges.csv", self.resolved.degraded_edges)
        (output / "summary.json").write_text(
            json.dumps(self.summary, indent=2)
        )
        write_csv(output / "od_impacts.csv", self.od_impacts)
        write_csv(output / "edge_rerouting.csv", self.edge_rerouting)
        (output / "affected_zones.geojson").write_text(
            json.dumps(self.affected_zones)
        )
        (output / "node_impacts.geojson").write_text(
            json.dumps(self.node_impacts)
        )
        (output / "route_examples.geojson").write_text(
            json.dumps(self.route_examples)
        )
        self._write_edge_geojson(output / "edge_rerouting.geojson")
        graph_to_geojson(
            self.resolved.graph, output / "disrupted_network.geojson"
        )

    def _write_edge_geojson(self, path: Path) -> None:
        features = []
        for row in self.edge_rerouting:
            first, second = row["source"], row["target"]
            graph = (
                self.resolved.graph
                if self.resolved.graph.has_edge(first, second)
                else self.baseline_graph
            )
            if not graph.has_edge(first, second):
                continue
            first_data, second_data = graph.nodes[first], graph.nodes[second]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [float(first_data["x"]), float(first_data["y"])],
                            [float(second_data["x"]), float(second_data["y"])],
                        ],
                    },
                    "properties": row,
                }
            )
        path.write_text(
            json.dumps({"type": "FeatureCollection", "features": features})
        )


class SimulationEngine:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.output = Path(config["paths"]["output"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.graph = nx.read_graphml(config["paths"]["graph"])
        self.demand = _read_demand(config["paths"]["demand"])
        assignment = config["assignment"]
        self.alpha = float(assignment["bpr_alpha"])
        self.beta = float(assignment["bpr_beta"])
        self.tolerance = float(assignment["tolerance"])
        self.max_iterations = int(assignment["max_iterations"])
        baseline_cache = config.get("baseline_cache")
        if baseline_cache:
            self.baseline_assignment, self.baseline_routes = (
                self._load_baseline_cache(Path(baseline_cache))
            )
        else:
            self.baseline_assignment = assign_msa(
                self.graph,
                self.demand,
                self.alpha,
                self.beta,
                self.max_iterations,
                self.tolerance,
            )
            self.baseline_routes = _route_records(
                self.graph,
                self.demand,
                self.baseline_assignment.edge_costs,
            )
        self.baseline_by_pair = {
            (row["origin"], row["destination"]): row
            for row in self.baseline_routes
        }
        if not baseline_cache:
            self.write_baseline_cache()

    @classmethod
    def from_config(cls, path: str | Path) -> "SimulationEngine":
        return cls(load_config(path))

    def write_baseline_cache(self) -> None:
        output = self.output / "baseline"
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        route_map = {}
        for row in self.baseline_routes:
            rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"route_nodes", "route_edges"}
                }
            )
            route_map[f"{row['origin']}->{row['destination']}"] = {
                "route_nodes": row["route_nodes"],
                "route_edges": row["route_edges"],
            }
        write_csv(output / "baseline_od_routes.csv", rows)
        with gzip.open(output / "baseline_routes.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(route_map, handle)
        edge_rows = self._edge_rows(
            self.graph, self.baseline_assignment.edge_flows
        )
        write_csv(output / "baseline_edge_state.csv", edge_rows)
        components = nx.number_connected_components(self.graph)
        largest = len(max(nx.connected_components(self.graph), key=len))
        served_routes = [row for row in self.baseline_routes if row["served"]]
        served_weight = sum(float(row["demand"]) for row in served_routes)
        mean_distance = sum(
            float(row["demand"]) * float(row["distance_m"])
            for row in served_routes
        ) / max(served_weight, 1e-12)
        summary = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "components": components,
            "largest_component_nodes": largest,
            "served_demand_ratio": self.baseline_assignment.served_demand_ratio,
            "average_shortest_path_length_m": mean_distance,
            "mean_travel_time_min": self.baseline_assignment.mean_travel_time,
            "global_efficiency": self.baseline_assignment.global_efficiency,
            "msa_iterations": self.baseline_assignment.iterations,
            "msa_relative_gap": self.baseline_assignment.convergence,
        }
        (output / "baseline_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

    def _load_baseline_cache(
        self, cache_dir: Path
    ) -> tuple[AssignmentResult, list[dict]]:
        required = (
            "baseline_summary.json",
            "baseline_od_routes.csv",
            "baseline_routes.json.gz",
            "baseline_edge_state.csv",
        )
        missing = [name for name in required if not (cache_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Baseline cache is incomplete: {', '.join(missing)}"
            )
        summary = json.loads(
            (cache_dir / "baseline_summary.json").read_text(encoding="utf-8")
        )
        if int(summary["nodes"]) != self.graph.number_of_nodes():
            raise ValueError("Baseline cache node count does not match graph")
        if int(summary["edges"]) != self.graph.number_of_edges():
            raise ValueError("Baseline cache edge count does not match graph")

        edge_rows = []
        with (cache_dir / "baseline_edge_state.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            edge_rows = list(csv.DictReader(handle))
        flows = {
            edge_key(row["source"], row["target"]): float(row["flow"])
            for row in edge_rows
        }
        network = SparseAssignmentNetwork(self.graph)
        flow_array = np.asarray(
            [flows.get(key, 0.0) for key in network.keys], dtype=float
        )
        cost_array = bpr_time(
            network.free_flow_time,
            flow_array,
            network.capacity,
            self.alpha,
            self.beta,
        )
        edge_costs = {
            key: float(value) for key, value in zip(network.keys, cost_array)
        }
        total_demand = sum(float(row["demand"]) for row in self.demand)
        served_demand = (
            float(summary["served_demand_ratio"]) * total_demand
        )
        assignment = AssignmentResult(
            edge_flows=flows,
            edge_costs=edge_costs,
            served_demand=served_demand,
            total_demand=total_demand,
            mean_travel_time=float(summary["mean_travel_time_min"]),
            global_efficiency=float(summary["global_efficiency"]),
            overloaded_edges=sum(
                float(row["volume_capacity_ratio"]) > 1.0
                for row in edge_rows
            ),
            iterations=int(summary["msa_iterations"]),
            convergence=float(summary["msa_relative_gap"]),
        )

        with gzip.open(
            cache_dir / "baseline_routes.json.gz", "rt", encoding="utf-8"
        ) as handle:
            route_map = json.load(handle)
        routes = []
        with (cache_dir / "baseline_od_routes.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                key = f"{row['origin']}->{row['destination']}"
                geometry = route_map.get(
                    key, {"route_nodes": [], "route_edges": []}
                )
                routes.append(
                    {
                        "origin": row["origin"],
                        "destination": row["destination"],
                        "demand": float(row["demand"]),
                        "served": str(row["served"]).lower() == "true",
                        "travel_time_min": _optional_float(
                            row["travel_time_min"]
                        ),
                        "distance_m": _optional_float(row["distance_m"]),
                        "route_nodes": geometry["route_nodes"],
                        "route_edges": [
                            tuple(value) for value in geometry["route_edges"]
                        ],
                    }
                )
        if len(routes) != len(self.demand):
            raise ValueError("Baseline route cache does not match demand rows")
        return assignment, routes

    def preview(self, scenario: ScenarioSpec | dict) -> SimulationResult:
        return self._run(scenario, "preview")

    def simulate(self, scenario: ScenarioSpec | dict) -> SimulationResult:
        return self._run(scenario, "exact")

    def _run(
        self, scenario: ScenarioSpec | dict, mode: str
    ) -> SimulationResult:
        if isinstance(scenario, dict):
            scenario = ScenarioSpec.from_dict(scenario)
        start = time.perf_counter()
        resolved = resolve_scenario(self.graph, scenario)
        if mode == "preview":
            assignment = _assignment_from_preview(
                resolved.graph,
                self.demand,
                self.baseline_assignment.edge_flows,
                self.alpha,
                self.beta,
            )
        else:
            assignment = assign_msa(
                resolved.graph,
                self.demand,
                self.alpha,
                self.beta,
                self.max_iterations,
                self.tolerance,
            )
        disrupted_routes = _route_records(
            resolved.graph, self.demand, assignment.edge_costs
        )
        (
            summary,
            od_impacts,
            edge_rerouting,
            affected_zones,
            node_impacts,
            route_examples,
        ) = self._analyze(
            scenario, mode, resolved, assignment, disrupted_routes
        )
        runtime = time.perf_counter() - start
        summary["runtime_seconds"] = runtime
        return SimulationResult(
            scenario,
            mode,
            resolved,
            assignment,
            summary,
            od_impacts,
            edge_rerouting,
            affected_zones,
            node_impacts,
            route_examples,
            runtime,
            self.graph,
        )

    def _analyze(
        self,
        scenario: ScenarioSpec,
        mode: str,
        resolved: ResolvedScenario,
        assignment: AssignmentResult,
        disrupted_routes: list[dict],
    ) -> tuple:
        threshold = float(self.config["impact"]["affected_threshold"])
        od_impacts = []
        served_demand = 0.0
        baseline_distance = disrupted_distance = 0.0
        baseline_time = disrupted_time = 0.0
        affected_demand = 0.0
        for disrupted in disrupted_routes:
            key = (disrupted["origin"], disrupted["destination"])
            baseline = self.baseline_by_pair[key]
            amount = float(disrupted["demand"])
            if disrupted["served"]:
                served_demand += amount
                baseline_distance += amount * float(baseline["distance_m"])
                disrupted_distance += amount * float(disrupted["distance_m"])
                baseline_time += amount * float(baseline["travel_time_min"])
                disrupted_time += amount * float(disrupted["travel_time_min"])
                distance_increase = (
                    float(disrupted["distance_m"])
                    / max(float(baseline["distance_m"]), 1e-9)
                    - 1.0
                )
                time_increase = (
                    float(disrupted["travel_time_min"])
                    / max(float(baseline["travel_time_min"]), 1e-9)
                    - 1.0
                )
                same_route = disrupted["route_edges"] == baseline["route_edges"]
                status = (
                    "unaffected"
                    if same_route and distance_increase < threshold and time_increase < threshold
                    else "rerouted"
                )
            else:
                distance_increase = time_increase = None
                status = "disconnected"
            affected = status == "disconnected" or (
                distance_increase is not None
                and (
                    distance_increase >= threshold
                    or time_increase >= threshold
                )
            )
            if affected:
                affected_demand += amount
            od_impacts.append(
                {
                    "origin": disrupted["origin"],
                    "destination": disrupted["destination"],
                    "demand": amount,
                    "baseline_distance_m": baseline["distance_m"],
                    "disrupted_distance_m": disrupted["distance_m"],
                    "baseline_time_min": baseline["travel_time_min"],
                    "disrupted_time_min": disrupted["travel_time_min"],
                    "distance_increase_ratio": distance_increase,
                    "time_increase_ratio": time_increase,
                    "status": status,
                    "affected": affected,
                    "baseline_route": json.dumps(baseline["route_nodes"]),
                    "disrupted_route": json.dumps(disrupted["route_nodes"]),
                }
            )
        total_demand = sum(float(row["demand"]) for row in self.demand)
        served_ratio = float(np.clip(served_demand / max(total_demand, 1e-12), 0, 1))
        path_resilience = float(
            np.clip(
                baseline_distance / max(disrupted_distance, 1e-12)
                if served_demand
                else 0.0,
                0,
                1,
            )
        )
        time_resilience = float(
            np.clip(
                baseline_time / max(disrupted_time, 1e-12)
                if served_demand
                else 0.0,
                0,
                1,
            )
        )
        service_adjusted = served_ratio * path_resilience
        components = (
            nx.number_connected_components(resolved.graph)
            if resolved.graph.number_of_nodes()
            else 0
        )
        largest = (
            len(max(nx.connected_components(resolved.graph), key=len))
            if resolved.graph.number_of_nodes()
            else 0
        )
        baseline_largest = len(max(nx.connected_components(self.graph), key=len))
        edge_rerouting = self._edge_impacts(
            resolved.graph, assignment.edge_flows
        )
        newly_overloaded = sum(row["newly_overloaded"] for row in edge_rerouting)
        positive_burden = sum(row["rerouting_burden"] for row in edge_rerouting)
        summary = {
            "scenario_id": scenario.scenario_id,
            "name": scenario.name,
            "hazard_type": scenario.hazard_type,
            "mode": mode,
            "closed_nodes": len(resolved.closed_nodes),
            "closed_edges": len(resolved.closed_edges),
            "degraded_edges": len(resolved.degraded_edges),
            "served_demand_ratio": served_ratio,
            "disconnected_demand_ratio": 1.0 - served_ratio,
            "path_resilience": path_resilience,
            "time_resilience": time_resilience,
            "service_adjusted_resilience": service_adjusted,
            "resilience_band": _resilience_band(service_adjusted),
            "path_length_increase": (
                max(
                    disrupted_distance / max(baseline_distance, 1e-12) - 1.0,
                    0.0,
                )
                if served_demand
                else None
            ),
            "travel_time_increase": (
                max(
                    disrupted_time / max(baseline_time, 1e-12) - 1.0,
                    0.0,
                )
                if served_demand
                else None
            ),
            "affected_demand_ratio": affected_demand / max(total_demand, 1e-12),
            "component_count": components,
            "component_count_increase": components
            - nx.number_connected_components(self.graph),
            "largest_component_loss": 1.0
            - largest / max(baseline_largest, 1),
            "global_efficiency": assignment.global_efficiency,
            "global_efficiency_loss": max(
                1.0
                - assignment.global_efficiency
                / max(self.baseline_assignment.global_efficiency, 1e-12),
                0.0,
            ),
            "newly_overloaded_edges": newly_overloaded,
            "total_positive_rerouting_burden": positive_burden,
            "msa_iterations": assignment.iterations,
            "msa_relative_gap": assignment.convergence,
            "converged": mode == "preview"
            or assignment.convergence <= self.tolerance,
        }
        affected_zones, node_impacts = self._affected_geojson(od_impacts, total_demand)
        route_examples = self._route_examples(od_impacts)
        return (
            summary,
            od_impacts,
            edge_rerouting,
            affected_zones,
            node_impacts,
            route_examples,
        )

    def _edge_rows(
        self, graph: nx.Graph, flows: dict[tuple[str, str], float]
    ) -> list[dict]:
        rows = []
        for first, second, data in graph.edges(data=True):
            key = edge_key(first, second)
            capacity = float(data["relative_capacity"])
            flow = flows.get(key, 0.0)
            rows.append(
                {
                    "source": str(first),
                    "target": str(second),
                    "flow": flow,
                    "relative_capacity": capacity,
                    "volume_capacity_ratio": flow / max(capacity, 1e-12),
                }
            )
        return rows

    def _edge_impacts(
        self, graph: nx.Graph, flows: dict[tuple[str, str], float]
    ) -> list[dict]:
        rows = []
        all_keys = set(self.baseline_assignment.edge_flows) | set(flows)
        for key in sorted(all_keys):
            baseline = self.baseline_assignment.edge_flows.get(key, 0.0)
            disrupted = flows.get(key, 0.0)
            if graph.has_edge(*key):
                capacity = float(graph[key[0]][key[1]]["relative_capacity"])
            elif self.graph.has_edge(*key):
                capacity = float(self.graph[key[0]][key[1]]["relative_capacity"])
            else:
                capacity = 1.0
            baseline_vc = baseline / max(
                float(self.graph[key[0]][key[1]]["relative_capacity"])
                if self.graph.has_edge(*key)
                else capacity,
                1e-12,
            )
            disrupted_vc = disrupted / max(capacity, 1e-12)
            rows.append(
                {
                    "source": key[0],
                    "target": key[1],
                    "baseline_flow": baseline,
                    "disrupted_flow": disrupted,
                    "flow_delta": disrupted - baseline,
                    "rerouting_burden": max(disrupted - baseline, 0.0),
                    "baseline_vc": baseline_vc,
                    "disrupted_vc": disrupted_vc,
                    "newly_overloaded": baseline_vc <= 1.0 < disrupted_vc,
                }
            )
        return rows

    def _affected_geojson(
        self, impacts: list[dict], total_demand: float
    ) -> tuple[dict, dict]:
        grid_size = float(self.config["impact"]["grid_size_m"])
        cells: dict[tuple[int, int], dict] = {}
        nodes: dict[str, dict] = {}
        for row in impacts:
            if not row["affected"]:
                continue
            half = float(row["demand"]) / 2.0
            for node_id in (row["origin"], row["destination"]):
                data = self.graph.nodes[node_id]
                x, y = float(data["x"]), float(data["y"])
                key = (math.floor(x / grid_size), math.floor(y / grid_size))
                cell = cells.setdefault(
                    key,
                    {
                        "affected_demand": 0.0,
                        "disconnected_demand": 0.0,
                        "path_weighted": 0.0,
                        "time_weighted": 0.0,
                    },
                )
                cell["affected_demand"] += half
                if row["status"] == "disconnected":
                    cell["disconnected_demand"] += half
                else:
                    cell["path_weighted"] += half * max(
                        float(row["distance_increase_ratio"]), 0
                    )
                    cell["time_weighted"] += half * max(
                        float(row["time_increase_ratio"]), 0
                    )
                node = nodes.setdefault(
                    node_id,
                    {"affected_demand": 0.0, "disconnected_demand": 0.0},
                )
                node["affected_demand"] += half
                if row["status"] == "disconnected":
                    node["disconnected_demand"] += half
        cell_features = []
        for (column, row), values in cells.items():
            affected = values["affected_demand"]
            path_mean = values["path_weighted"] / max(affected, 1e-12)
            time_mean = values["time_weighted"] / max(affected, 1e-12)
            impact = (
                affected / max(total_demand, 1e-12)
                + values["disconnected_demand"] / max(total_demand, 1e-12)
                + min(path_mean, 1.0)
                + min(time_mean, 1.0)
            )
            x0, y0 = column * grid_size, row * grid_size
            cell_features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [x0, y0],
                                [x0 + grid_size, y0],
                                [x0 + grid_size, y0 + grid_size],
                                [x0, y0 + grid_size],
                                [x0, y0],
                            ]
                        ],
                    },
                    "properties": {
                        **values,
                        "mean_path_increase": path_mean,
                        "mean_time_increase": time_mean,
                        "impact_score": impact,
                    },
                }
            )
        node_features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(self.graph.nodes[node_id]["x"]),
                        float(self.graph.nodes[node_id]["y"]),
                    ],
                },
                "properties": {"node_id": node_id, **values},
            }
            for node_id, values in nodes.items()
        ]
        return (
            {"type": "FeatureCollection", "features": cell_features},
            {"type": "FeatureCollection", "features": node_features},
        )

    def _route_examples(self, impacts: list[dict]) -> dict:
        def score(row: dict) -> float:
            if row["status"] == "disconnected":
                return 1e6 + float(row["demand"])
            return float(row["demand"]) * (
                max(float(row["distance_increase_ratio"] or 0), 0)
                + max(float(row["time_increase_ratio"] or 0), 0)
            )

        selected = sorted(impacts, key=score, reverse=True)[
            : int(self.config["impact"]["route_examples"])
        ]
        features = []
        for row in selected:
            for route_type, field in (
                ("baseline", "baseline_route"),
                ("disrupted", "disrupted_route"),
            ):
                node_ids = json.loads(row[field])
                if len(node_ids) < 2:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [
                                [
                                    float(self.graph.nodes[node]["x"]),
                                    float(self.graph.nodes[node]["y"]),
                                ]
                                for node in node_ids
                            ],
                        },
                        "properties": {
                            "origin": row["origin"],
                            "destination": row["destination"],
                            "route_type": route_type,
                            "status": row["status"],
                        },
                    }
                )
        return {"type": "FeatureCollection", "features": features}


def _resilience_band(value: float) -> str:
    if value >= 0.8:
        return "resilient"
    if value >= 0.6:
        return "moderate degradation"
    if value >= 0.4:
        return "vulnerable"
    if value >= 0.2:
        return "severe disruption"
    return "systemic failure"


def _optional_float(value: str) -> float | None:
    if value in {"", "None", "null"}:
        return None
    return float(value)
