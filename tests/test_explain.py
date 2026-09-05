"""
Unit tests for Module 6: Explainability Layer (SHAP).
"""

import unittest
from pathlib import Path

from src.explain.explainer import FraudExplainer, ShapExplanation
from src.models.train import MODELS_DIR


class TestExplainability(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.explainer = FraudExplainer(models_dir=MODELS_DIR)

    def test_explainer_initialization(self):
        self.assertIsNotNone(self.explainer.model)
        self.assertIsNotNone(self.explainer.scaler)
        self.assertEqual(len(self.explainer.feature_names), 36)

    def test_explain_single_transaction(self):
        # Sample test transaction feature dictionary
        sample_feats = {name: 0.0 for name in self.explainer.feature_names}
        sample_feats["amount"] = 850.0
        sample_feats["spending_deviation"] = 12.5
        sample_feats["V4"] = 3.8
        sample_feats["V12"] = -4.5

        explanation = self.explainer.explain_transaction(sample_feats, transaction_id="tx_test_explain")
        self.assertIsInstance(explanation, ShapExplanation)
        self.assertEqual(explanation.transaction_id, "tx_test_explain")
        self.assertGreaterEqual(explanation.prediction_prob, 0.0)
        self.assertLessEqual(explanation.prediction_prob, 1.0)
        self.assertGreater(len(explanation.top_contributors), 0)
        self.assertIn("tx_test_explain", explanation.summary_reason)

        # Verify ascii waterfall rendering
        chart_str = explanation.render_ascii_waterfall(top_k=5)
        self.assertIn("SHAP ATTRIBUTION", chart_str)
        self.assertIn("P(Fraud)", chart_str)


if __name__ == "__main__":
    unittest.main()
