"""
Module 5: Statistical Drift Detection Engine.

Implements Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI)
to detect data distribution shifts and prediction drift between baseline training
distributions and recent incoming streaming transactions.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.models.train import MODELS_DIR


@dataclass
class FeatureDriftResult:
    """Drift metrics for an individual feature."""
    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    status: str            # 'STABLE', 'WARNING', 'ALERT'
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float


@dataclass
class DriftReport:
    """Batch-level drift detection summary."""
    timestamp: str
    batch_size: int
    overall_status: str    # 'STABLE', 'DRIFT_DETECTED'
    drifted_features_count: int
    total_features_monitored: int
    feature_results: Dict[str, FeatureDriftResult]
    prediction_drift_psi: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["feature_results"] = {k: asdict(v) for k, v in self.feature_results.items()}
        return res


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) between reference (expected) and current (actual).
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Filter out NaNs or infs
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    # Calculate bin counts
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions with smoothing epsilon
    expected_pct = np.maximum(expected_counts / len(expected), epsilon)
    actual_pct = np.maximum(actual_counts / len(actual), epsilon)

    # Re-normalize
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()

    # Sum PSI components
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi_val))


def calculate_ks(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test.
    Returns (statistic, p_value).
    """
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, 1.0
    res = stats.ks_2samp(expected, actual)
    return float(res.statistic), float(res.pvalue)


class DriftMonitor:
    """
    Monitors incoming batches against a baseline reference dataset using KS and PSI.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        psi_warning_threshold: float = 0.10,
        psi_alert_threshold: float = 0.20,
        ks_p_value_threshold: float = 0.01,
        reference_predictions: Optional[np.ndarray] = None,
    ):
        self.reference_data = reference_data.copy()
        self.reference_predictions = reference_predictions
        self.psi_warning = psi_warning_threshold
        self.psi_alert = psi_alert_threshold
        self.ks_p_thresh = ks_p_value_threshold

    def evaluate_batch(
        self,
        current_data: pd.DataFrame,
        current_predictions: Optional[np.ndarray] = None,
        features_to_monitor: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Runs KS-test and PSI for continuous features in current_data against reference_data.
        """
        import datetime

        cols = features_to_monitor or [
            c for c in current_data.columns
            if c in self.reference_data.columns and np.issubdtype(current_data[c].dtype, np.number)
        ]

        feature_results = {}
        drifted_count = 0

        for col in cols:
            ref_vals = self.reference_data[col].values
            curr_vals = current_data[col].values

            psi_val = calculate_psi(ref_vals, curr_vals)
            ks_stat, ks_p = calculate_ks(ref_vals, curr_vals)

            # Determine alert status
            if psi_val >= self.psi_alert or ks_p < self.ks_p_thresh:
                status = "ALERT"
                drifted_count += 1
            elif psi_val >= self.psi_warning:
                status = "WARNING"
            else:
                status = "STABLE"

            feature_results[col] = FeatureDriftResult(
                feature_name=col,
                psi=round(psi_val, 4),
                ks_statistic=round(ks_stat, 4),
                ks_p_value=round(ks_p, 6),
                status=status,
                reference_mean=round(float(np.nanmean(ref_vals)), 4),
                current_mean=round(float(np.nanmean(curr_vals)), 4),
                reference_std=round(float(np.nanstd(ref_vals)), 4),
                current_std=round(float(np.nanstd(curr_vals)), 4),
            )

        # Optional prediction score drift
        pred_psi = None
        if self.reference_predictions is not None and current_predictions is not None:
            pred_psi = round(calculate_psi(self.reference_predictions, current_predictions), 4)

        overall_status = "DRIFT_DETECTED" if drifted_count > 0 else "STABLE"

        return DriftReport(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            batch_size=len(current_data),
            overall_status=overall_status,
            drifted_features_count=drifted_count,
            total_features_monitored=len(cols),
            feature_results=feature_results,
            prediction_drift_psi=pred_psi,
        )


def create_synthetic_drifted_batch(
    base_df: pd.DataFrame,
    shift_features: Optional[List[str]] = None,
    amount_multiplier: float = 3.5,
    noise_shift: float = 2.5,
) -> pd.DataFrame:
    """
    Creates an intentionally drifted batch to test drift detection:
    - Multiplies Amount by amount_multiplier (simulating inflation or large-ticket attack).
    - Shifts key PCA features by noise_shift std deviations.
    """
    drifted_df = base_df.copy()
    targets = shift_features or ["amount", "rolling_avg_spend_1h", "V1", "V4", "V12", "spending_deviation"]

    for col in targets:
        if col in drifted_df.columns:
            if "amount" in col or "spend" in col:
                drifted_df[col] = drifted_df[col] * amount_multiplier
            else:
                std = np.nanstd(drifted_df[col]) or 1.0
                drifted_df[col] = drifted_df[col] + (noise_shift * std)

    return drifted_df
