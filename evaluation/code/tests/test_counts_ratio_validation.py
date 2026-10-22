import unittest

import pandas as pd

from code.counts_ratio_validation import (
    add_case_flags,
    cancellation_fraction,
    safe_correlation,
    temporal_segment_ids,
)


class CountsRatioValidationTests(unittest.TestCase):
    def test_cancellation_fraction_distinguishes_net_from_gross_error(self) -> None:
        self.assertAlmostEqual(cancellation_fraction(pd.Series([0.2, -0.2])), 1.0)
        self.assertAlmostEqual(cancellation_fraction(pd.Series([0.2, 0.2])), 0.0)
        self.assertAlmostEqual(cancellation_fraction(pd.Series([0.0, 0.0])), 0.0)

    def test_ratio_close_can_coexist_with_count_and_classification_error(self) -> None:
        frame = pd.DataFrame(
            {
                "count_error": [-4, 0],
                "ratio_absolute_error": [0.02, 0.2],
                "final_matched_classification_errors": [2, 0],
                "gt_shape_classification_errors": [1, 0],
            }
        )
        result = add_case_flags(frame, 0.05)
        self.assertTrue(result.loc[0, "count_wrong_ratio_close"])
        self.assertTrue(result.loc[0, "final_class_wrong_ratio_close"])
        self.assertTrue(result.loc[0, "gt_shape_class_wrong_ratio_close"])
        self.assertFalse(result.loc[1, "ratio_close"])

    def test_temporal_segments_cover_every_frame_once(self) -> None:
        segments = temporal_segment_ids(69)
        counts = pd.Series(segments).value_counts().to_dict()
        self.assertEqual(counts, {"early": 23, "middle": 23, "late": 23})

    def test_constant_series_has_no_meaningful_correlation(self) -> None:
        value = safe_correlation(pd.Series([1, 1, 1]), pd.Series([1, 2, 3]), "pearson")
        self.assertIsNone(value)


if __name__ == "__main__":
    unittest.main()
