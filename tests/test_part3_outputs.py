from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

import networkx as nx


class Part3OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = Path("runs/part3")
        cls.summary = json.loads(
            (cls.output / "part3_summary.json").read_text()
        )
        cls.graph = nx.read_graphml(cls.output / "transport_graph.graphml")

    def test_transport_graph_attributes(self) -> None:
        self.assertEqual(nx.number_of_selfloops(self.graph), 0)
        for _, _, data in self.graph.edges(data=True):
            self.assertGreater(float(data["length_m"]), 0)
            self.assertGreater(float(data["relative_capacity"]), 0)
            self.assertGreater(float(data["speed_kmh"]), 0)
            self.assertGreater(float(data["free_flow_time_min"]), 0)

    def test_consolidation_safety_gate(self) -> None:
        consolidation = self.summary["consolidation"]
        self.assertLessEqual(
            consolidation["selected"]["false_bridge_rate"],
            consolidation["false_bridge_limit"],
        )
        self.assertFalse(consolidation["coverage_gate_met"])
        self.assertEqual(
            consolidation["routing_policy"], "largest_safe_component"
        )

    def test_baseline_assignments_converged(self) -> None:
        self.assertTrue(self.summary["gravity_baseline"]["converged"])
        self.assertTrue(self.summary["uniform_sensitivity"]["converged"])

    def test_resilience_bounds_and_targeted_damage(self) -> None:
        with (self.output / "resilience_curves.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_key = {
            (row["strategy"], row["removal_percentage"]): float(
                row["resilience_index"]
            )
            for row in rows
        }
        for value in by_key.values():
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
        for percentage in ("0.01", "0.02", "0.05", "0.1"):
            random_value = by_key[("random", percentage)]
            targeted = min(
                by_key[(strategy, percentage)]
                for strategy in (
                    "highest_degree",
                    "highest_betweenness",
                    "highest_flow_criticality",
                )
            )
            self.assertLessEqual(targeted, random_value)


if __name__ == "__main__":
    unittest.main()
