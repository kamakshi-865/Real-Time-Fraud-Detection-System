"""
Unit tests for Module 4: Retraining Pipeline & Model Versioning.
"""

import unittest
import tempfile
import json
from pathlib import Path

from src.models.retrain import ModelRegistry, RetrainingPipeline
from data.setup_data import DEFAULT_CSV_PATH


class TestRetraining(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name)
        self.registry = ModelRegistry(models_dir=self.models_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_version_progression(self):
        self.assertEqual(self.registry.get_next_version(), "v1")
        # Fake v1 artifact
        (self.models_dir / "model_v1.pkl").touch()
        self.assertEqual(self.registry.get_next_version(), "v2")
        # Fake v2 artifact
        (self.models_dir / "model_v2.pkl").touch()
        self.assertEqual(self.registry.get_next_version(), "v3")

    def test_trigger_retrain_creates_versioned_artifacts(self):
        pipeline = RetrainingPipeline(csv_path=DEFAULT_CSV_PATH, models_dir=self.models_dir)
        
        # Trigger first retrain (v1)
        res1 = pipeline.trigger_retrain(max_rows=400, reason="initial_monthly_batch", target_version="v1")
        self.assertEqual(res1["version"], "v1")
        self.assertTrue(Path(res1["model_path"]).exists())
        self.assertTrue(Path(res1["metadata_path"]).exists())

        # Trigger second retrain (v2)
        res2 = pipeline.trigger_retrain(max_rows=500, reason="new_monthly_data_arrived")
        self.assertEqual(res2["version"], "v2")
        self.assertEqual(res2["parent_version"], "v1")
        self.assertTrue(Path(res2["model_path"]).exists())
        self.assertTrue(Path(res2["metadata_path"]).exists())

        # Check metadata json fields
        with open(res2["metadata_path"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["version"], "v2")
        self.assertEqual(meta["trigger_reason"], "new_monthly_data_arrived")
        self.assertIn("metrics", meta)
        self.assertIn("data_summary", meta)


if __name__ == "__main__":
    unittest.main()
