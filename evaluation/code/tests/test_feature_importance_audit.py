import unittest

import pandas as pd

from code.feature_importance_audit import high_correlation_pairs, summarize_runs


class FeatureImportanceAuditTests(unittest.TestCase):
    def test_high_correlations_are_unique_upper_triangle_pairs(self) -> None:
        frame = pd.DataFrame(
            [
                {"split": "training", "feature_a": "area", "feature_b": "area", "spearman_correlation": 1.0, "absolute_spearman_correlation": 1.0},
                {"split": "training", "feature_a": "area", "feature_b": "perimeter", "spearman_correlation": 0.91, "absolute_spearman_correlation": 0.91},
                {"split": "training", "feature_a": "perimeter", "feature_b": "area", "spearman_correlation": 0.91, "absolute_spearman_correlation": 0.91},
                {"split": "training", "feature_a": "area", "feature_b": "eccentricity", "spearman_correlation": 0.2, "absolute_spearman_correlation": 0.2},
            ]
        )
        selected = high_correlation_pairs(frame)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["feature_a"], "area")
        self.assertEqual(selected.iloc[0]["feature_b"], "perimeter")

    def test_run_summary_separates_shuffle_and_between_run_variation(self) -> None:
        runs = pd.DataFrame(
            [
                {"seed": 1, "feature": "area", "importance_mean": 0.2, "rank": 1},
                {"seed": 2, "feature": "area", "importance_mean": 0.4, "rank": 1},
            ]
        )
        draws = pd.DataFrame(
            [
                {"seed": 1, "feature": "area", "importance_drop": 0.1},
                {"seed": 1, "feature": "area", "importance_drop": 0.3},
                {"seed": 2, "feature": "area", "importance_drop": 0.3},
                {"seed": 2, "feature": "area", "importance_drop": 0.5},
            ]
        )
        summary = summarize_runs(runs, draws, "feature").iloc[0]
        self.assertAlmostEqual(summary["importance_mean"], 0.3)
        self.assertAlmostEqual(summary["between_run_std"], (0.02) ** 0.5)
        self.assertEqual(summary["permutations"], 4)


if __name__ == "__main__":
    unittest.main()
