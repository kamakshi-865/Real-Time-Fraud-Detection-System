"""
Unit tests for Module 2: Feature Engineering & Feature Store.
"""

import unittest
import tempfile
import os
from pathlib import Path

from src.ingestion.streamer import TransactionEvent
from src.features.pipeline import FeatureEngineer, EnrichedFeatureVector
from db.storage import FeatureStoreDB


class TestFeaturePipeline(unittest.TestCase):

    def setUp(self):
        # Create a temporary SQLite database for test isolation
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db_file.name
        self.temp_db_file.close()

        self.db = FeatureStoreDB(db_path=self.temp_db_path)
        self.engineer = FeatureEngineer(db_path=self.temp_db_path, window_seconds=3600.0)

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_single_transaction_enrichment_and_storage(self):
        event = TransactionEvent(
            transaction_id="tx_test_001",
            timestamp="2026-09-06 01:00:00",
            simulated_seconds=100.0,
            user_id="usr_0001",
            amount=50.0,
            features={"V1": 0.5, "V2": -1.2},
            label=0
        )

        enriched = self.engineer.process_transaction(event, persist=True)
        self.assertEqual(enriched.transaction_id, "tx_test_001")
        self.assertEqual(enriched.user_id, "usr_0001")
        self.assertEqual(enriched.time_since_last_tx, 86400.0) # first transaction default
        self.assertEqual(enriched.tx_count_1h, 0)
        self.assertEqual(enriched.rolling_avg_spend_1h, 50.0)

        # Retrieve from SQLite feature store
        retrieved = self.engineer.get_feature_vector("tx_test_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["transaction_id"], "tx_test_001")
        self.assertEqual(retrieved["amount"], 50.0)
        self.assertEqual(retrieved["raw_features"]["V1"], 0.5)

    def test_sequential_user_velocity_and_rolling_aggregations(self):
        # Transaction 1: $100 at t=100s
        tx1 = TransactionEvent(
            transaction_id="tx_01",
            timestamp="2026-09-06 01:00:00",
            simulated_seconds=100.0,
            user_id="usr_repeat",
            amount=100.0,
            features={"V1": 0.1},
            label=0
        )
        self.engineer.process_transaction(tx1)

        # Transaction 2: $200 at t=400s (300s later, within 1h)
        tx2 = TransactionEvent(
            transaction_id="tx_02",
            timestamp="2026-09-06 01:05:00",
            simulated_seconds=400.0,
            user_id="usr_repeat",
            amount=200.0,
            features={"V1": 0.2},
            label=0
        )
        enr2 = self.engineer.process_transaction(tx2)

        self.assertEqual(enr2.time_since_last_tx, 300.0)
        self.assertEqual(enr2.tx_count_1h, 1)
        self.assertEqual(enr2.rolling_avg_spend_1h, 100.0) # Avg of prior tx
        self.assertGreater(enr2.spending_deviation, 0.0) # $200 exceeds $100 baseline

        # Transaction 3: $300 at t=700s
        tx3 = TransactionEvent(
            transaction_id="tx_03",
            timestamp="2026-09-06 01:10:00",
            simulated_seconds=700.0,
            user_id="usr_repeat",
            amount=300.0,
            features={"V1": 0.3},
            label=1
        )
        enr3 = self.engineer.process_transaction(tx3)

        self.assertEqual(enr3.time_since_last_tx, 300.0)
        self.assertEqual(enr3.tx_count_1h, 2)
        self.assertEqual(enr3.rolling_avg_spend_1h, 150.0) # (100 + 200)/2
        self.assertEqual(enr3.rolling_spend_sum_1h, 300.0)


if __name__ == "__main__":
    unittest.main()
