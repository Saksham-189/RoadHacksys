from __future__ import annotations

import unittest

import torch

from roadseg.legacy_metrics import final_score as legacy_final_score
from roadseg.metrics import metrics_from_counts, relaxed_counts
from roadseg.model import SegFormer


class PipelineTests(unittest.TestCase):
    def test_model_output_shape(self) -> None:
        model = SegFormer()
        output = model(torch.randn(1, 4, 256, 256))
        self.assertEqual(output.shape, (1, 1, 256, 256))

    def test_perfect_metrics(self) -> None:
        metrics = metrics_from_counts(10, 0, 0, 20)
        self.assertAlmostEqual(metrics["iou"], 1.0)
        self.assertAlmostEqual(metrics["dice"], 1.0)

    def test_relaxed_metric_accepts_small_shift(self) -> None:
        target = torch.zeros(1, 1, 16, 16)
        prediction = torch.zeros_like(target)
        target[:, :, :, 7] = 1
        prediction[:, :, :, 8] = 1
        strict_intersection, _, _ = relaxed_counts(prediction, target, radius=0)
        relaxed_intersection, _, _ = relaxed_counts(prediction, target, radius=1)
        self.assertEqual(strict_intersection, 0)
        self.assertGreater(relaxed_intersection, 0)

    def test_historical_e012_score_reproduces_report(self) -> None:
        clean = {
            "iou": 0.20579830165245788,
            "dice": 0.34134780480355037,
            "recall": 0.817788917709961,
            "connectivity_ratio": 0.8711244557436052,
            "relaxed_iou": 0.27010085972786546,
        }
        occluded = {"occlusion_recall": 0.9179072016000727}
        self.assertAlmostEqual(
            legacy_final_score(clean, occluded), 0.5733576366, places=6
        )


if __name__ == "__main__":
    unittest.main()
