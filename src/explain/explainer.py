"""
Module 6: Explainability Layer using SHAP (SHapley Additive exPlanations).

Provides local per-transaction feature attributions and global feature importance.
Explains exactly which features drove a transaction to be flagged as fraud,
powering auditor review workflows and compliance transparency.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.models.retrain import ModelRegistry, MODELS_DIR


@dataclass
class FeatureContribution:
    """Individual feature's impact on model prediction."""
    feature_name: str
    shap_value: float
    feature_value: float
    impact: str          # 'INCREASES_FRAUD_RISK', 'DECREASES_FRAUD_RISK'
    magnitude: float


@dataclass
class ShapExplanation:
    """Complete interpretability summary for a single transaction."""
    transaction_id: str
    prediction_prob: float
    is_fraud_flagged: bool
    base_value: float
    top_contributors: List[FeatureContribution]
    summary_reason: str

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["top_contributors"] = [asdict(c) for c in self.top_contributors]
        return res

    def render_ascii_waterfall(self, top_k: int = 8) -> str:
        """Renders an ASCII horizontal bar chart representing SHAP attribution."""
        lines = []
        lines.append(f"Base Value E[f(x)]: {self.base_value:+.4f} | Prediction P(Fraud): {self.prediction_prob*100:.2f}%")
        lines.append("-" * 75)
        lines.append(f"{'FEATURE':<22} | {'VAL':>8} | {'SHAP ATTRIBUTION':>16} | {'RISK DIRECTION'}")
        lines.append("-" * 75)

        for c in self.top_contributors[:top_k]:
            bar_len = min(20, int(abs(c.shap_value) * 12))
            if c.shap_value > 0:
                bar = f"[+{c.shap_value:.3f}] " + ("#" * max(1, bar_len))
                dir_str = "--> +FRAUD RISK"
            else:
                bar = f"[-{abs(c.shap_value):.3f}] " + ("." * max(1, bar_len))
                dir_str = "<-- -LEGIT SHIELD"

            lines.append(f"{c.feature_name:<22} | {c.feature_value:>8.2f} | {bar:<16} | {dir_str}")

        lines.append("-" * 75)
        lines.append(f"Summary: {self.summary_reason}")
        return "\n".join(lines)


class FraudExplainer:
    """
    SHAP-based interpretability engine for real-time inference explanation.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.registry = ModelRegistry(models_dir=self.models_dir)
        self.model, self.scaler, self.metadata = self.registry.get_active_model()
        self.feature_names = (
            self.metadata.get("feature_names")
            or self.metadata.get("features")
            or getattr(self.scaler, "feature_names_in_", []).tolist()
        )

        # Initialize TreeExplainer for XGBoost (sub-millisecond evaluation)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_transaction(
        self,
        feature_dict: Dict[str, float],
        transaction_id: str = "tx_sample",
        threshold: float = 0.5,
    ) -> ShapExplanation:
        """
        Computes exact SHAP attributions for an individual transaction.
        """
        # Order features matching model's expectation
        row_vals = [feature_dict.get(col, 0.0) for col in self.feature_names]
        X_raw = np.array([row_vals], dtype=float)

        # Scale features using model's scaler (preserve feature names)
        df_raw = pd.DataFrame(X_raw, columns=self.feature_names)
        X_scaled = self.scaler.transform(df_raw)

        # Predict probability
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        is_flagged = prob >= threshold

        # Compute SHAP values
        shap_raw = self.explainer.shap_values(X_scaled)
        if isinstance(shap_raw, list):
            # Binary classification in some shap versions returns list of [class 0, class 1]
            shap_values = shap_raw[1][0]
        elif len(shap_raw.shape) == 2:
            shap_values = shap_raw[0]
        else:
            shap_values = shap_raw

        base_val = float(self.explainer.expected_value if not isinstance(self.explainer.expected_value, (list, np.ndarray)) else self.explainer.expected_value[1])

        contributions = []
        for name, val, s_val in zip(self.feature_names, row_vals, shap_values):
            s_val = float(s_val)
            impact = "INCREASES_FRAUD_RISK" if s_val > 0 else "DECREASES_FRAUD_RISK"
            contributions.append(FeatureContribution(
                feature_name=name,
                shap_value=round(s_val, 4),
                feature_value=round(float(val), 4),
                impact=impact,
                magnitude=round(abs(s_val), 4),
            ))

        # Sort by absolute magnitude of contribution
        contributions.sort(key=lambda c: c.magnitude, reverse=True)

        # Natural language explanation
        top_positive = [c for c in contributions if c.shap_value > 0][:3]
        if is_flagged:
            reasons = [f"{c.feature_name} ({c.feature_value:.2f}, SHAP {c.shap_value:+.3f})" for c in top_positive]
            summary = f"Transaction {transaction_id}: Flagged as FRAUD (Risk Score: {prob*100:.1f}%). Primary risk drivers: {', '.join(reasons)}."
        else:
            summary = f"Transaction {transaction_id}: Assessed as LEGITIMATE (Risk Score: {prob*100:.1f}%). Normal behavioral patterns."

        return ShapExplanation(
            transaction_id=transaction_id,
            prediction_prob=round(prob, 4),
            is_fraud_flagged=is_flagged,
            base_value=round(base_val, 4),
            top_contributors=contributions,
            summary_reason=summary,
        )

    def explain_batch_summary(self, df_features: pd.DataFrame) -> Dict[str, float]:
        """
        Global feature importance: Mean absolute SHAP values across a batch.
        """
        X_raw = df_features[self.feature_names].values
        X_scaled = self.scaler.transform(X_raw)
        shap_vals = self.explainer.shap_values(X_scaled)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        importance = {name: float(round(score, 4)) for name, score in zip(self.feature_names, mean_abs)}
        return dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
