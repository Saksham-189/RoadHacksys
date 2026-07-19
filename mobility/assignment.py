from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass
class AssignmentResult:
    edge_flows: dict[tuple[str, str], float]
    edge_costs: dict[tuple[str, str], float]
    served_demand: float
    total_demand: float
    mean_travel_time: float
    global_efficiency: float
    overloaded_edges: int
    iterations: int
    convergence: float

    @property
    def served_demand_ratio(self) -> float:
        return float(
            np.clip(
                self.served_demand / max(self.total_demand, 1e-12),
                0.0,
                1.0,
            )
        )


def edge_key(first: object, second: object) -> tuple[str, str]:
    values = sorted((str(first), str(second)))
    return values[0], values[1]


def bpr_time(
    free_flow_time: np.ndarray,
    flow: np.ndarray,
    capacity: np.ndarray,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> np.ndarray:
    ratio = np.divide(flow, capacity, out=np.zeros_like(flow), where=capacity > 0)
    return free_flow_time * (1.0 + alpha * np.power(ratio, beta))


class SparseAssignmentNetwork:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.node_index = {str(node): index for index, node in enumerate(self.nodes)}
        self.edges = [(str(first), str(second)) for first, second in graph.edges]
        self.keys = [edge_key(first, second) for first, second in self.edges]
        self.key_index = {key: index for index, key in enumerate(self.keys)}
        self.first_indices = np.asarray(
            [self.node_index[first] for first, _ in self.edges], dtype=np.int32
        )
        self.second_indices = np.asarray(
            [self.node_index[second] for _, second in self.edges], dtype=np.int32
        )
        self.free_flow_time = np.asarray(
            [
                float(graph[first][second]["free_flow_time_min"])
                for first, second in self.edges
            ],
            dtype=np.float64,
        )
        self.capacity = np.asarray(
            [
                float(graph[first][second]["relative_capacity"])
                for first, second in self.edges
            ],
            dtype=np.float64,
        )

    def matrix(self, costs: np.ndarray) -> csr_matrix:
        rows = np.concatenate((self.first_indices, self.second_indices))
        columns = np.concatenate((self.second_indices, self.first_indices))
        values = np.concatenate((costs, costs))
        return csr_matrix(
            (values, (rows, columns)),
            shape=(len(self.nodes), len(self.nodes)),
        )

    def all_or_nothing(
        self, demand: list[dict], costs: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        flows = np.zeros(len(self.edges), dtype=np.float64)
        valid = [
            row
            for row in demand
            if str(row["origin"]) in self.node_index
            and str(row["destination"]) in self.node_index
        ]
        total_demand = float(sum(float(row["demand"]) for row in demand))
        if not valid:
            return flows, {
                "served_demand": 0.0,
                "total_demand": total_demand,
                "mean_travel_time": 0.0,
                "global_efficiency": 0.0,
            }
        origins = sorted({str(row["origin"]) for row in valid})
        origin_positions = {origin: index for index, origin in enumerate(origins)}
        origin_indices = [self.node_index[origin] for origin in origins]
        distances, predecessors = dijkstra(
            self.matrix(costs),
            directed=False,
            indices=origin_indices,
            return_predecessors=True,
        )
        served = 0.0
        weighted_time = 0.0
        weighted_efficiency = 0.0
        for row in valid:
            origin = str(row["origin"])
            destination = str(row["destination"])
            amount = float(row["demand"])
            source_index = self.node_index[origin]
            destination_index = self.node_index[destination]
            source_row = origin_positions[origin]
            travel_time = float(distances[source_row, destination_index])
            if not np.isfinite(travel_time):
                continue
            current = destination_index
            path_edges = []
            while current != source_index:
                previous = int(predecessors[source_row, current])
                if previous < 0:
                    path_edges = []
                    break
                key = edge_key(self.nodes[previous], self.nodes[current])
                edge_index = self.key_index.get(key)
                if edge_index is None:
                    path_edges = []
                    break
                path_edges.append(edge_index)
                current = previous
            if not path_edges and source_index != destination_index:
                continue
            flows[path_edges] += amount
            served += amount
            weighted_time += amount * travel_time
            weighted_efficiency += amount / max(travel_time, 1e-9)
        return flows, {
            "served_demand": served,
            "total_demand": total_demand,
            "mean_travel_time": weighted_time / max(served, 1e-12),
            "global_efficiency": weighted_efficiency / max(total_demand, 1e-12),
        }


def assign_msa(
    graph: nx.Graph,
    demand: list[dict],
    alpha: float,
    beta: float,
    max_iterations: int,
    tolerance: float,
) -> AssignmentResult:
    network = SparseAssignmentNetwork(graph)
    flows = np.zeros(len(network.edges), dtype=np.float64)
    convergence = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        costs = bpr_time(
            network.free_flow_time, flows, network.capacity, alpha, beta
        )
        auxiliary, shortest_metrics = network.all_or_nothing(demand, costs)
        system_cost = float(np.dot(flows, costs))
        shortest_cost = float(
            shortest_metrics["mean_travel_time"]
            * shortest_metrics["served_demand"]
        )
        if system_cost > 0:
            convergence = max(
                0.0, (system_cost - shortest_cost) / system_cost
            )
        step = 1.0 / iteration
        updated = flows + step * (auxiliary - flows)
        flows = updated
        if iteration > 1 and convergence <= tolerance:
            break
    final_costs = bpr_time(
        network.free_flow_time, flows, network.capacity, alpha, beta
    )
    _, metrics = network.all_or_nothing(demand, final_costs)
    ratios = np.divide(
        flows,
        network.capacity,
        out=np.zeros_like(flows),
        where=network.capacity > 0,
    )
    return AssignmentResult(
        edge_flows={
            key: float(flow) for key, flow in zip(network.keys, flows)
        },
        edge_costs={
            key: float(cost) for key, cost in zip(network.keys, final_costs)
        },
        served_demand=float(metrics["served_demand"]),
        total_demand=float(metrics["total_demand"]),
        mean_travel_time=float(metrics["mean_travel_time"]),
        global_efficiency=float(metrics["global_efficiency"]),
        overloaded_edges=int((ratios > 1.0).sum()),
        iterations=iteration,
        convergence=convergence,
    )


def scale_demand_to_target(
    graph: nx.Graph,
    demand: list[dict],
    target_p90_vc: float,
) -> tuple[list[dict], float]:
    network = SparseAssignmentNetwork(graph)
    flows, _ = network.all_or_nothing(demand, network.free_flow_time)
    ratios = np.divide(
        flows,
        network.capacity,
        out=np.zeros_like(flows),
        where=network.capacity > 0,
    )
    positive = ratios[ratios > 0]
    current = float(np.percentile(positive, 90)) if positive.size else 1.0
    factor = target_p90_vc / max(current, 1e-12)
    scaled = [{**row, "demand": float(row["demand"]) * factor} for row in demand]
    return scaled, factor
