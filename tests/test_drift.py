"""
Unit tests for Module 5: Statistical Drift Detection Engine (KS-Test & PSI).
"""

import unittest
import numpy as np
import pandas as pd

from src.monitoring.drift import (
    calculate_psi,
    calculate_ks,
    DriftMonitor,
    create_synthetic_drifted_batch,
)


class TestDriftDetection(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 1000
        self.ref_data = pd.DataFrame({
            "amount": np.random.lognormal(3.0, 1.0, n),
            "V1": np.random.normal(0.0, 1.0, n),
            "V4": np.random.normal(0.0, 1.0, n),
            "spending_deviation": np.random.normal(0.0, 1.0, n),
        })

    def test_psi_identical_vs_shifted(self):
        # Identical distribution sample
        sample_identical = np.random.normal(0.0, 1.0, 1000)
        ref_sample = np.random.normal(0.0, 1.0, 1000)
        psi_low = calculate_psi(ref_sample, sample_identical)
        self.assertLess(psi_low, 0.10) # Stable

        # Heavily shifted distribution (mean shifted by +2.0)
        sample_shifted = np.random.normal(2.0, 1.0, 1000)
        psi_high = calculate_psi(ref_sample, sample_shifted)
        self.assertGreater(psi_high, 0.25) # Critical alert

    def test_ks_test_p_values(self):
        ref_sample = np.random.normal(0.0, 1.0, 1000)
        sample_identical = np.random.normal(0.0, 1.0, 1000)
        stat, p_val = calculate_ks(ref_sample, sample_identical)
        self.assertGreater(p_val, 0.01)

        sample_shifted = np.random.normal(1.5, 1.0, 1000)
        stat_drift, p_drift = calculate_ks(ref_sample, sample_shifted)
        self.assertLess(p_drift, 0.001)

    def test_monitor_stable_vs_drifted_batch(self):
        monitor = DriftMonitor(self.ref_data)

        # 1. Normal in-distribution batch
        normal_batch = pd.DataFrame({
            "amount": np.random.lognormal(3.0, 1.0, 300),
            "V1": np.random.normal(0.0, 1.0, 300),
            "V4": np.random.normal(0.0, 1.0, 300),
            "spending_deviation": np.random.normal(0.0, 1.0, 300),
        })
        rep_normal = monitor.evaluate_batch(normal_batch)
        self.assertEqual(rep_normal.overall_status, "STABLE")
        self.assertEqual(rep_normal.drifted_features_count, 0)

        # 2. Intentionally drifted batch
        drifted_batch = create_synthetic_drifted_batch(normal_batch, amount_multiplier=4.0, noise_shift=3.0)
        rep_drift = monitor.evaluate_batch(drifted_batch)
        self.assertEqual(rep_drift.overall_status, "DRIFT_DETECTED")
        self.assertGreater(rep_drift.drifted_features_count, 0)
        self.assertEqual(rep_drift.feature_results["amount"].status, "ALERT")


if __name__ == "__main__":
    unittest.main()
