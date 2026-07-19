from __future__ import annotations

import unittest

import numpy as np

from roadgraph.benchmark import component_statistics
from roadgraph.extract import graph_from_mask, skeleton_from_mask
from roadgraph.healing import DisjointSet, EXPERIMENTS, heal_mask


class RoadGraphTests(unittest.TestCase):
    def test_cross_becomes_graph(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[16, 4:28] = True
        mask[4:28, 16] = True
        graph = graph_from_mask(mask, resolution_m=10)
        self.assertGreaterEqual(len(graph.nodes), 5)
        self.assertGreaterEqual(len(graph.edges), 4)

    def test_disjoint_set_rejects_cycle(self) -> None:
        sets = DisjointSet([1, 2, 3])
        self.assertTrue(sets.union(1, 2))
        self.assertTrue(sets.union(2, 3))
        self.assertFalse(sets.union(1, 3))

    def test_hybrid_connects_aligned_gap(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        mask[32, 8:28] = True
        mask[32, 36:56] = True
        probability = mask.astype(np.float32)
        probability[32, 28:36] = 0.4
        before = component_statistics(mask)[0]
        healed, bridges = heal_mask(mask, probability, EXPERIMENTS["B007"])
        after = component_statistics(skeleton_from_mask(healed))[0]
        self.assertEqual(before, 2)
        self.assertEqual(after, 1)
        self.assertGreaterEqual(len(bridges), 1)


if __name__ == "__main__":
    unittest.main()
