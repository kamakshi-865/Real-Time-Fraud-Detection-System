"""
Unit tests for Module 3: Model Training, Imbalance Handling, & Evaluation.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from src.models.train import ModelTrainer, EvaluationReport
from data.setup_data import DEFAULT_CSV_PATH


class TestModelTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.trainer = ModelTrainer(DEFAULT_CSV_PATH, models_dir=Path(cls.temp_dir.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_dataset_preparation_and_scaling(self):
        X_train, X_test, y_train, y_test, scaler = self.trainer.prepare_dataset(max_rows=400, test_size=0.25)
        self.assertEqual(len(X_train), 300)
        self.assertEqual(len(X_test), 100)
        self.assertGreater(len(self.trainer.feature_names), 30)
        self.assertEqual(X_train.shape[1], len(self.trainer.feature_names))

    def test_evaluation_metrics_contract(self):
        X_train, X_test, y_train, y_test, scaler = self.trainer.prepare_dataset(max_rows=400, test_size=0.25)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X_train, y_train)

        report = self.trainer.evaluate_model(clf, X_test, y_test, "TestLogReg", "None")
        self.assertIsInstance(report, EvaluationReport)
        self.assertGreaterEqual(report.pr_auc, 0.0)
        self.assertLessEqual(report.pr_auc, 1.0)
        self.assertEqual(report.tn + report.fp + report.fn + report.tp, len(y_test))

    def test_save_champion_model(self):
        X_train, X_test, y_train, y_test, scaler = self.trainer.prepare_dataset(max_rows=300, test_size=0.25)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(X_train, y_train)
        report = self.trainer.evaluate_model(clf, X_test, y_test, "TestLogReg", "None")

        m_path, meta_path = self.trainer.save_champion_model(clf, scaler, report, version="test_v1")
        self.assertTrue(m_path.exists())
        self.assertTrue(meta_path.exists())


if __name__ == "__main__":
    unittest.main()
