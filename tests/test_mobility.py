from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from mobility.assignment import assign_msa, bpr_time, edge_key
from mobility.centrality import betweenness_reference, degree_baseline
from mobility.stress import consequence_metrics


def add_edge(
    graph: nx.Graph,
    first: str,
    second: str,
    time: float = 1.0,
    capacity: float = 10.0,
) -> None:
    graph.add_edge(
        first,
        second,
        length_m=100.0,
        relative_capacity=capacity,
        speed_kmh=30.0,
        road_class="medium",
        free_flow_time_min=time,
        travel_time_min=time,
    )


def set_coordinates(graph: nx.Graph) -> None:
    for index, node in enumerate(graph.nodes):
        graph.nodes[node].update(x=float(index * 100), y=0.0, row=0.0, col=float(index))


class MobilityTests(unittest.TestCase):
    def test_bpr_time_is_monotonic(self) -> None:
        free = np.array([1.0, 1.0, 1.0])
        flow = np.array([0.0, 5.0, 10.0])
        capacity = np.array([10.0, 10.0, 10.0])
        times = bpr_time(free, flow, capacity)
        self.assertLess(times[0], times[1])
        self.assertLess(times[1], times[2])

    def test_star_center_wins_degree_baseline(self) -> None:
        graph = nx.star_graph(5)
        graph = nx.relabel_nodes(graph, str)
        for first, second in graph.edges:
            graph[first][second].update(relative_capacity=1.0)
        set_coordinates(graph)
        self.assertEqual(degree_baseline(graph)[0]["node_id"], "0")

    def test_line_middle_has_betweenness_and_articulation(self) -> None:
        graph = nx.path_graph(["0", "1", "2", "3", "4"])
        for first, second in graph.edges:
            graph[first][second]["free_flow_time_min"] = 1.0
        set_coordinates(graph)
        nodes, _ = betweenness_reference(graph, 5, 42)
        self.assertEqual(nodes[0]["node_id"], "2")
        self.assertTrue(nodes[0]["articulation_point"])

    def test_msa_conserves_flow_on_path(self) -> None:
        graph = nx.Graph()
        add_edge(graph, "0", "1")
        add_edge(graph, "1", "2")
        set_coordinates(graph)
        demand = [{"origin": "0", "destination": "2", "demand": 4.0}]
        result = assign_msa(graph, demand, 0.15, 4.0, 20, 1e-6)
        self.assertAlmostEqual(result.edge_flows[edge_key("0", "1")], 4.0)
        self.assertAlmostEqual(result.edge_flows[edge_key("1", "2")], 4.0)
        self.assertAlmostEqual(result.served_demand_ratio, 1.0)

    def test_msa_prefers_faster_parallel_route(self) -> None:
        graph = nx.Graph()
        add_edge(graph, "o", "a", time=1.0, capacity=20.0)
        add_edge(graph, "a", "d", time=1.0, capacity=20.0)
        add_edge(graph, "o", "b", time=2.0, capacity=20.0)
        add_edge(graph, "b", "d", time=2.0, capacity=20.0)
        set_coordinates(graph)
        demand = [{"origin": "o", "destination": "d", "demand": 10.0}]
        result = assign_msa(graph, demand, 0.15, 4.0, 30, 1e-5)
        fast = result.edge_flows[edge_key("o", "a")]
        slow = result.edge_flows[edge_key("o", "b")]
        self.assertGreater(fast, slow)

    def test_bridge_removal_reduces_resilience(self) -> None:
        graph = nx.path_graph(["0", "1", "2"])
        for first, second in graph.edges:
            graph[first][second].update(
                free_flow_time_min=1.0, relative_capacity=10.0
            )
        set_coordinates(graph)
        demand = [{"origin": "0", "destination": "2", "demand": 2.0}]
        baseline = assign_msa(graph, demand, 0.15, 4.0, 10, 1e-6)
        perturbed_graph = graph.copy()
        perturbed_graph.remove_edge("1", "2")
        perturbed = assign_msa(
            perturbed_graph, demand, 0.15, 4.0, 10, 1e-6
        )
        metrics = consequence_metrics(
            baseline, perturbed, perturbed_graph, len(graph)
        )
        self.assertEqual(metrics["served_demand_ratio"], 0.0)
        self.assertEqual(metrics["resilience_index"], 0.0)


if __name__ == "__main__":
    unittest.main()
