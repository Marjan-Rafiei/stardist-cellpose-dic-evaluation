from __future__ import annotations

import unittest

from code.segmentation_analytics_error_chain import (
    build_count_decomposition,
    build_ratio_bridge,
)


class SegmentationAnalyticsErrorChainTests(unittest.TestCase):
    def test_total_count_decomposition_reaches_prediction_total(self) -> None:
        rows = build_count_decomposition(
            system="example",
            gt_total=20,
            false_negatives=6,
            false_positives=3,
            gt_mitotic=5,
            false_negative_mitotic=2,
            false_positive_mitotic=1,
            baseline_classifier_net_mitotic=-1,
            predicted_shape_net_mitotic=2,
        )
        total_rows = [row for row in rows if row["quantity"] == "total_count"]
        self.assertEqual(total_rows[-1]["running_total"], 17)

    def test_mitotic_count_decomposition_is_additive(self) -> None:
        rows = build_count_decomposition(
            system="example",
            gt_total=20,
            false_negatives=6,
            false_positives=3,
            gt_mitotic=5,
            false_negative_mitotic=2,
            false_positive_mitotic=1,
            baseline_classifier_net_mitotic=-1,
            predicted_shape_net_mitotic=2,
        )
        mitotic_rows = [row for row in rows if row["quantity"] == "mitotic_count"]
        self.assertEqual(mitotic_rows[-1]["running_total"], 5)

    def test_ratio_bridge_reaches_observed_prediction_ratio(self) -> None:
        rows = build_ratio_bridge(
            system="example",
            gt_mitotic=5,
            gt_total=20,
            matched_gt_mitotic=3,
            matched_total=14,
            false_positive_mitotic=1,
            false_positives=3,
            gt_shape_classifier_mitotic=2,
            pred_shape_classifier_mitotic=4,
        )
        self.assertAlmostEqual(rows[-1]["mitotic_ratio"], 5 / 17)
        self.assertEqual(rows[-1]["total_denominator"], 17)


if __name__ == "__main__":
    unittest.main()
