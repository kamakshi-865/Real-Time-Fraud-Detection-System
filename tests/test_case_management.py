"""
Unit tests for Module 7: Case Management & Reviewer Decision Storage.
"""

import unittest
import tempfile
import os
from pathlib import Path

from db.storage import FeatureStoreDB


class TestCaseManagement(unittest.TestCase):

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.db = FeatureStoreDB(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_flagged_case_lifecycle_and_reviewer_decision(self):
        # 1. Save flagged case
        case_payload = {
            "transaction_id": "tx_case_001",
            "user_id": "usr_9999",
            "timestamp": "2026-09-06 01:50:00",
            "amount": 950.00,
            "risk_score": 0.985,
            "status": "PENDING",
            "shap_summary": "High risk detected on amount and V4",
            "shap_features": [
                {"feature_name": "amount", "shap_value": 2.1, "feature_value": 950.0},
                {"feature_name": "V4", "shap_value": 1.8, "feature_value": 4.5},
            ]
        }
        self.db.save_flagged_case(case_payload)

        # 2. Verify pending retrieval
        pending = self.db.get_flagged_cases(status_filter="PENDING")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["transaction_id"], "tx_case_001")
        self.assertEqual(pending[0]["status"], "PENDING")
        self.assertEqual(len(pending[0]["shap_features"]), 2)

        stats_before = self.db.get_reviewer_stats()
        self.assertEqual(stats_before["pending"], 1)
        self.assertEqual(stats_before["confirmed_fraud"], 0)

        # 3. Record reviewer triage action: Confirmed Fraud
        self.db.record_reviewer_decision(
            transaction_id="tx_case_001",
            decision="CONFIRMED_FRAUD",
            notes="Cardholder confirmed stolen credential."
        )

        # 4. Verify status updated
        pending_after = self.db.get_flagged_cases(status_filter="PENDING")
        self.assertEqual(len(pending_after), 0)

        resolved = self.db.get_flagged_cases(status_filter="CONFIRMED_FRAUD")
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["status"], "CONFIRMED_FRAUD")

        stats_after = self.db.get_reviewer_stats()
        self.assertEqual(stats_after["pending"], 0)
        self.assertEqual(stats_after["confirmed_fraud"], 1)

        # 5. Check audit trail
        history = self.db.get_decision_history(limit=5)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["transaction_id"], "tx_case_001")
        self.assertEqual(history[0]["decision"], "CONFIRMED_FRAUD")
        self.assertEqual(history[0]["notes"], "Cardholder confirmed stolen credential.")


if __name__ == "__main__":
    unittest.main()
