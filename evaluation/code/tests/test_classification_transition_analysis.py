from __future__ import annotations

import unittest

from code.classification_transition_analysis import transition_label


class ClassificationTransitionAnalysisTests(unittest.TestCase):
    def test_all_four_transition_labels(self) -> None:
        expected = {
            (True, True): "correct_to_correct",
            (True, False): "correct_to_incorrect",
            (False, False): "incorrect_to_incorrect",
            (False, True): "incorrect_to_correct",
        }
        for values, label in expected.items():
            with self.subTest(values=values):
                self.assertEqual(transition_label(*values), label)

    def test_incorrect_to_correct_is_not_collapsed_into_correct(self) -> None:
        self.assertNotEqual(
            transition_label(False, True),
            transition_label(True, True),
        )


if __name__ == "__main__":
    unittest.main()
