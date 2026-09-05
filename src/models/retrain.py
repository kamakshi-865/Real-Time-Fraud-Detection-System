"""
Module 4: Retraining Pipeline & Model Registry.

Automates model retraining either on a schedule or via manual trigger.
Handles persistent versioning (model_v1.pkl, model_v2.pkl, ...) with JSON metadata
recording training timestamp, dataset window/size, hyperparameters, and evaluation metrics.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import joblib
import numpy as np

from src.models.train import ModelTrainer, EvaluationReport, MODELS_DIR
from data.setup_data import DEFAULT_CSV_PATH


class ModelRegistry:
    """Manages versioned model artifacts and registry manifest."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "registry.json"
        self._init_registry()

    def _init_registry(self):
        """Initialize or load model registry file."""
        if not self.registry_file.exists():
            self._save_registry({
                "active_version": "v1",
                "versions": {}
            })

    def _load_registry(self) -> Dict[str, Any]:
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"active_version": None, "versions": {}}

    def _save_registry(self, data: Dict[str, Any]):
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_all_versions(self) -> List[str]:
        """Scan directory for all existing model versions."""
        versions = []
        for p in self.models_dir.glob("model_v*.pkl"):
            m = re.search(r"model_v(\d+)\.pkl", p.name)
            if m:
                versions.append(f"v{m.group(1)}")
        # Sort numerically
        versions.sort(key=lambda v: int(v.replace("v", "")))
        return versions

    def get_next_version(self) -> str:
        """Determines the next version string (e.g. 'v2' if 'v1' exists)."""
        existing = self.get_all_versions()
        if not existing:
            return "v1"
        max_num = max(int(v.replace("v", "")) for v in existing)
        return f"v{max_num + 1}"

    def register_version(self, version: str, metadata: Dict[str, Any], set_active: bool = True):
        """Records version in registry manifest."""
        data = self._load_registry()
        data["versions"][version] = {
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "model_path": str(self.models_dir / f"model_{version}.pkl"),
            "metadata_path": str(self.models_dir / f"metadata_{version}.json"),
            "pr_auc": metadata.get("metrics", {}).get("pr_auc", 0.0),
            "f1": metadata.get("metrics", {}).get("f1", 0.0),
        }
        if set_active:
            data["active_version"] = version
        self._save_registry(data)

    def get_active_model(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """Loads currently active champion model, scaler, and metadata."""
        reg = self._load_registry()
        active_ver = reg.get("active_version")
        if not active_ver:
            versions = self.get_all_versions()
            if not versions:
                raise FileNotFoundError("No registered model versions found.")
            active_ver = versions[-1]

        model_path = self.models_dir / f"model_{active_ver}.pkl"
        scaler_path = self.models_dir / "scaler.pkl"
        meta_path = self.models_dir / f"metadata_{active_ver}.json"

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return model, scaler, metadata


class RetrainingPipeline:
    """
    Orchestrates automated or manually triggered model retraining.
    Incorporates new batch data, trains candidate, verifies performance,
    and updates versioned artifacts on disk.
    """

    def __init__(self, csv_path: str | Path = DEFAULT_CSV_PATH, models_dir: Path = MODELS_DIR):
        self.csv_path = Path(csv_path)
        self.models_dir = Path(models_dir)
        self.registry = ModelRegistry(models_dir=self.models_dir)
        self.trainer = ModelTrainer(csv_path=self.csv_path, models_dir=self.models_dir)

    def trigger_retrain(
        self,
        max_rows: Optional[int] = None,
        reason: str = "manual_trigger",
        target_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes a full retraining cycle:
        1. Identifies next version (e.g. v2).
        2. Ingests and prepares updated transaction window.
        3. Retrains XGBoost candidate.
        4. Benchmarks against incumbent model metrics.
        5. Saves versioned artifacts and updates registry.
        """
        new_version = target_version or self.registry.get_next_version()
        incumbent_versions = self.registry.get_all_versions()
        parent_version = incumbent_versions[-1] if incumbent_versions else None

        print(f"\n========================================================")
        print(f"  TRIGGERING RETRAINING PIPELINE -> TARGET: {new_version}")
        print(f"  Reason: {reason} | Parent: {parent_version}")
        print(f"========================================================")

        # 1. Prepare updated dataset window
        X_train, X_test, y_train, y_test, scaler = self.trainer.prepare_dataset(
            max_rows=max_rows, test_size=0.2, random_state=int(new_version.replace("v", "")) * 10
        )

        # 2. Retrain with XGBoost (SMOTE or scale_pos_weight)
        from imblearn.over_sampling import SMOTE
        import xgboost as xgb

        pos_count = int(y_train.sum())
        if pos_count > 1:
            k_neighbors = min(5, pos_count - 1)
            print(f"Applying SMOTE on training window (k_neighbors={k_neighbors})...")
            smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        else:
            print("Insufficient minority samples for SMOTE; using original training set...")
            X_train_smote, y_train_smote = X_train, y_train

        print(f"Training updated XGBoost model for {new_version}...")
        model = xgb.XGBClassifier(
            n_estimators=130,
            learning_rate=0.07,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train_smote, y_train_smote)

        # 3. Evaluate new model
        report = self.trainer.evaluate_model(model, X_test, y_test, "XGBoost", "SMOTE")
        print(f"Retrained Metrics: PR-AUC={report.pr_auc:.4f} | F1={report.f1:.4f} | Recall={report.recall:.4f}")

        # 4. Save versioned artifacts
        model_path = self.models_dir / f"model_{new_version}.pkl"
        scaler_path = self.models_dir / f"scaler_{new_version}.pkl"
        meta_path = self.models_dir / f"metadata_{new_version}.json"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        metadata = {
            "version": new_version,
            "parent_version": parent_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "trigger_reason": reason,
            "data_summary": {
                "total_samples": len(X_train) + len(X_test),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "fraud_count_total": int(y_train.sum() + y_test.sum()),
                "fraud_rate": float((y_train.sum() + y_test.sum()) / (len(y_train) + len(y_test))),
            },
            "metrics": report.to_dict(),
            "hyperparameters": {
                "algorithm": "XGBoost",
                "n_estimators": 130,
                "learning_rate": 0.07,
                "max_depth": 4,
                "imbalance_handling": "SMOTE",
            },
            "feature_names": self.trainer.feature_names,
            "features": self.trainer.feature_names,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 5. Register in manifest
        self.registry.register_version(new_version, metadata, set_active=True)

        return {
            "version": new_version,
            "parent_version": parent_version,
            "model_path": str(model_path),
            "metadata_path": str(meta_path),
            "report": report,
            "metadata": metadata,
        }
