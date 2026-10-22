from __future__ import annotations

import unittest

import numpy as np

from code.temporal_bootstrap_audit import autocorrelation, sample_indices


class TemporalBootstrapAuditTests(unittest.TestCase):
    def test_autocorrelation_lag_zero_is_one(self) -> None:
        values = np.asarray([0.0, 1.0, 3.0, 2.0, 5.0])
        result = autocorrelation(values, max_lag=3)
        self.assertAlmostEqual(float(result[0]), 1.0)

    def test_autocorrelation_matches_direct_formula(self) -> None:
        values = np.asarray([1.0, 2.0, 4.0, 3.0, 6.0])
        centered = values - values.mean()
        expected = np.dot(centered[1:], centered[:-1]) / np.dot(centered, centered)
        self.assertAlmostEqual(float(autocorrelation(values, max_lag=2)[1]), float(expected))

    def test_block_sample_is_deterministic_and_has_frame_length(self) -> None:
        first = sample_indices(np.random.default_rng(42), n_frames=69, block_length=10)
        second = sample_indices(np.random.default_rng(42), n_frames=69, block_length=10)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 69)

    def test_block_sample_rejects_invalid_length(self) -> None:
        with self.assertRaises(ValueError):
            sample_indices(np.random.default_rng(42), n_frames=69, block_length=70)


if __name__ == "__main__":
    unittest.main()
