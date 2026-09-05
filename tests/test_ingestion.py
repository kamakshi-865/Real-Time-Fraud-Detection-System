"""
Unit tests for Module 1: Data Ingestion & Streaming Engine.
"""

import unittest
from pathlib import Path
from src.ingestion.streamer import TransactionStreamer, QueueStreamer, StreamSchemaConfig

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


class TestStreamer(unittest.TestCase):

    def setUp(self):
        self.assertTrue(DATA_PATH.exists(), f"Dataset missing at {DATA_PATH}")

    def test_generator_streaming_limit(self):
        streamer = TransactionStreamer(DATA_PATH, delay_seconds=0.0)
        events = list(streamer.stream(max_events=15))
        self.assertEqual(len(events), 15)
        first_event = events[0]
        self.assertTrue(first_event.transaction_id.startswith("tx_"))
        self.assertTrue(first_event.user_id.startswith("usr_"))
        self.assertGreaterEqual(first_event.amount, 0.0)
        self.assertIn("V1", first_event.features)
        self.assertIn("V28", first_event.features)

    def test_queue_streamer_decoupled(self):
        q_streamer = QueueStreamer(DATA_PATH, delay_seconds=0.001)
        q_streamer.start(max_events=10)
        received = []
        while True:
            evt = q_streamer.get(timeout=2.0)
            if evt is None:
                break
            received.append(evt)
        q_streamer.stop()
        self.assertEqual(len(received), 10)


if __name__ == "__main__":
    unittest.main()
