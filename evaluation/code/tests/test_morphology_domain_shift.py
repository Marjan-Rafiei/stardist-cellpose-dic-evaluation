from __future__ import annotations

import unittest

import numpy as np

from code.morphology_domain_shift import summarize_feature


class MorphologyDomainShiftTests(unittest.TestCase):
    def test_summary_uses_paired_differences(self) -> None:
        result = summarize_feature(
            system="example",
            feature="area",
            gt_values=np.asarray([10.0, 20.0, 40.0, 80.0]),
            pred_values=np.asarray([12.0, 24.0, 48.0, 96.0]),
        )
        self.assertAlmostEqual(result["median_relative_difference_percent"], 20.0)
        self.assertAlmostEqual(result["positive_fraction"], 1.0)
        self.assertTrue(result["descriptively_systematic"])

    def test_mixed_directions_are_not_flagged_systematic(self) -> None:
        result = summarize_feature(
            system="example",
            feature="area",
            gt_values=np.asarray([10.0, 20.0, 30.0, 40.0]),
            pred_values=np.asarray([11.0, 19.0, 32.0, 38.0]),
        )
        self.assertAlmostEqual(result["positive_fraction"], 0.5)
        self.assertAlmostEqual(result["negative_fraction"], 0.5)
        self.assertFalse(result["descriptively_systematic"])

    def test_nonpositive_ground_truth_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_feature(
                system="example",
                feature="area",
                gt_values=np.asarray([0.0, 1.0]),
                pred_values=np.asarray([1.0, 2.0]),
            )


if __name__ == "__main__":
    unittest.main()
