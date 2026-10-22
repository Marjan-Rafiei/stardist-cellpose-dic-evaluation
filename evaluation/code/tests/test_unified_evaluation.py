import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from code.train_classifier import collect_rows, validate_split
from code.unified_evaluation import (
    DetectionCounts,
    detection_counts,
    detection_metrics,
    end_to_end_class_counts,
    extract_features,
    confusion_metrics,
    macro_summary,
    match_iou_matrix,
    pooled_metrics,
)


class UnifiedEvaluationTests(unittest.TestCase):
    def test_perfect_match(self):
        truth = np.array([[0, 1, 1], [0, 0, 2], [0, 0, 2]])
        prediction = np.array([[0, 7, 7], [0, 0, 9], [0, 0, 9]])
        counts, matches = detection_counts(truth, prediction, 0.5)
        self.assertEqual(counts, DetectionCounts(2, 0, 0, 2.0, 2))
        self.assertEqual(detection_metrics(counts)["f1"], 1.0)
        self.assertEqual(len(matches), 2)

    def test_both_empty_is_perfect_by_convention(self):
        empty = np.zeros((3, 3), dtype=int)
        counts, _ = detection_counts(empty, empty, 0.5)
        metrics = detection_metrics(counts)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["pq"], 1.0)
        self.assertIsNone(metrics["matched_iou"])

    def test_false_positive_only_and_false_negative_only(self):
        empty = np.zeros((3, 3), dtype=int)
        one = empty.copy()
        one[1, 1] = 4
        fp_counts, _ = detection_counts(empty, one)
        fn_counts, _ = detection_counts(one, empty)
        self.assertEqual((fp_counts.tp, fp_counts.fp, fp_counts.fn), (0, 1, 0))
        self.assertEqual((fn_counts.tp, fn_counts.fp, fn_counts.fn), (0, 0, 1))
        self.assertEqual(detection_metrics(fp_counts)["f1"], 0.0)
        self.assertEqual(detection_metrics(fn_counts)["f1"], 0.0)

    def test_merge_and_split_are_penalized(self):
        truth = np.array([[1, 1, 0, 2, 2]])
        merged = np.array([[9, 9, 9, 9, 9]])
        merge_counts, _ = detection_counts(truth, merged, 0.3)
        self.assertEqual((merge_counts.tp, merge_counts.fp, merge_counts.fn), (1, 0, 1))

        one_truth = np.array([[1, 1, 1, 1]])
        split = np.array([[7, 7, 8, 8]])
        split_counts, _ = detection_counts(one_truth, split, 0.5)
        self.assertEqual((split_counts.tp, split_counts.fp, split_counts.fn), (1, 1, 0))

    def test_hungarian_assignment_beats_greedy_counterexample(self):
        matrix = np.array([[0.90, 0.80], [0.85, 0.10]])
        matches = match_iou_matrix([1, 2], [10, 20], matrix, threshold=0.5)
        self.assertEqual({(match.gt_id, match.pred_id) for match in matches}, {(1, 20), (2, 10)})

    def test_threshold_aware_assignment_maximizes_valid_match_count(self):
        # Unconstrained max-IoU assignment chooses the diagonal (1.48 total)
        # and retains only one pair after thresholding.  The required
        # lexicographic assignment chooses two eligible off-diagonal pairs.
        matrix = np.array([[0.99, 0.51], [0.50, 0.49]])
        matches = match_iou_matrix([1, 2], [10, 20], matrix, threshold=0.5)
        self.assertEqual({(match.gt_id, match.pred_id) for match in matches}, {(1, 20), (2, 10)})

    def test_class_swap_counts_as_fp_and_fn_for_opposite_classes(self):
        gt_mask = np.array([[1, 1, 0, 2, 2]])
        gt_classes = np.array([[1, 1, 0, 2, 2]])
        pred_mask = np.array([[7, 7, 0, 8, 8]])
        _, matches = detection_counts(gt_mask, pred_mask, 0.5)
        counts = end_to_end_class_counts(
            matches,
            [1, 2],
            [7, 8],
            gt_mask,
            gt_classes,
            {7: 2, 8: 1},
        )
        self.assertEqual(counts[1], {"tp": 0, "fp": 1, "fn": 1})
        self.assertEqual(counts[2], {"tp": 0, "fp": 1, "fn": 1})

    def test_pooled_and_per_frame_macro_are_explicitly_different(self):
        counts = [DetectionCounts(1, 0, 0, 1.0, 1), DetectionCounts(1, 9, 0, 1.0, 1)]
        rows = pd.DataFrame([detection_metrics(row) for row in counts])
        self.assertTrue(np.isclose(macro_summary(rows)["f1"]["mean"], 13 / 22))
        self.assertTrue(np.isclose(pooled_metrics(counts)["f1"], 4 / 13))

    def test_tiny_object_feature_extraction_is_finite(self):
        mask = np.zeros((3, 3), dtype=int)
        mask[1, 1] = 1
        features = extract_features(mask)
        self.assertEqual(len(features), 1)
        self.assertTrue(np.isfinite(features.drop(columns="instance_id").to_numpy()).all())
        self.assertEqual(features.loc[0, "aspect_ratio"], 0.0)

    def test_confusion_metrics_reports_accuracy_balanced_accuracy_and_macro_f1(self):
        metrics = confusion_metrics(np.array([[8, 2], [1, 9]]))
        self.assertTrue(np.isclose(metrics["accuracy"], 0.85))
        self.assertTrue(np.isclose(metrics["balanced_accuracy"], 0.85))
        self.assertIn("macro_f1", metrics)

    def test_split_guard_rejects_mi05(self):
        validate_split(("MI01", "MI02", "MI03"), "MI04")
        with self.assertRaises(ValueError):
            validate_split(("MI01", "MI02", "MI05"), "MI04")

    def test_classifier_rows_require_an_aligned_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("Images", "Masks_converted", "Masks_classes"):
                (root / "MI01" / directory).mkdir(parents=True)
            image = np.zeros((3, 3), dtype=np.uint8)
            instance = np.zeros((3, 3), dtype=np.uint16)
            instance[1, 1] = 1
            classes = np.zeros((3, 3), dtype=np.uint8)
            classes[1, 1] = 1
            Image.fromarray(image).save(root / "MI01" / "Images" / "I_MI01_0002.png")
            for frame in (1, 2):
                Image.fromarray(instance).save(root / "MI01" / "Masks_converted" / f"M_MI01_{frame:04d}.png")
                Image.fromarray(classes).save(root / "MI01" / "Masks_classes" / f"M_MI01_{frame:04d}.png")
            rows = collect_rows(root, ("MI01",))
            self.assertEqual(rows["frame_id"].tolist(), ["MI01_0002"])


if __name__ == "__main__":
    unittest.main()
