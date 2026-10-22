from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from code.random_forest_audit import evaluate_predictions, feature_distributions


class RandomForestAuditTests(unittest.TestCase):
    def test_benchmark_metrics_use_macro_f1(self) -> None:
        result = evaluate_predictions(
            "example",
            np.asarray([1, 1, 2, 2]),
            np.asarray([1, 2, 2, 2]),
        )
        self.assertEqual(result["cm_interphase_as_mitotic"], 1)
        self.assertEqual(result["cm_mitotic_as_mitotic"], 2)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)

    def test_feature_distribution_is_class_specific(self) -> None:
        frame = pd.DataFrame(
            {
                "target": [1, 1, 2],
                "area": [1.0, 3.0, 10.0],
                "perimeter": [1.0, 3.0, 10.0],
                "eccentricity": [1.0, 3.0, 10.0],
                "solidity": [1.0, 3.0, 10.0],
                "extent": [1.0, 3.0, 10.0],
                "major_axis_length": [1.0, 3.0, 10.0],
                "minor_axis_length": [1.0, 3.0, 10.0],
                "aspect_ratio": [1.0, 3.0, 10.0],
            }
        )
        rows = feature_distributions(frame, "example")
        area_interphase = next(
            row for row in rows if row["feature"] == "area" and row["class_id"] == 1
        )
        area_mitotic = next(
            row for row in rows if row["feature"] == "area" and row["class_id"] == 2
        )
        self.assertEqual(area_interphase["median"], 2.0)
        self.assertEqual(area_mitotic["median"], 10.0)


if __name__ == "__main__":
    unittest.main()
