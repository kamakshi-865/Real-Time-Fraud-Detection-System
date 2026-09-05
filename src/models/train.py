"""
Module 3: Model Training & Evaluation Engine.

Implements baseline models (Logistic Regression, Random Forest) and XGBoost.
Explicitly addresses class imbalance using both SMOTE and Cost-Sensitive Class Weighting.
Evaluates models using Precision, Recall, F1, PR-AUC, and Confusion Matrix.
Includes analytical demonstration of why raw Accuracy is deceptive in fraud detection.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from src.ingestion.streamer import TransactionStreamer, StreamSchemaConfig
from src.features.pipeline import FeatureEngineer

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationReport:
    """Standardized performance metrics container."""
    model_name: str
    imbalance_method: str
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    accuracy: float
    confusion_matrix: List[List[int]]
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelTrainer:
    """
    End-to-End training and benchmarking pipeline for fraud detection models.
    Supports feature enrichment, SMOTE, class weighting, and metric reporting.
    """

    def __init__(self, csv_path: str | Path, models_dir: Path = MODELS_DIR):
        self.csv_path = Path(csv_path)
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names: List[str] = []

    def prepare_dataset(
        self,
        max_rows: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
        """
        Ingests transactions, applies FeatureEngineer for all 36 features,
        and produces stratified train/test splits.
        """
        print(f"Preparing dataset from {self.csv_path.name}...")
        streamer = TransactionStreamer(self.csv_path, delay_seconds=0.0)
        engineer = FeatureEngineer(window_seconds=3600.0)

        rows = []
        labels = []

        for evt in streamer.stream(max_events=max_rows):
            # Enrich with real-time features without persisting to DB
            enriched = engineer.process_transaction(evt, persist=False)
            feat_row = enriched.get_feature_matrix_row()
            rows.append(feat_row)
            labels.append(evt.label if evt.label is not None else 0)

        df_X = pd.DataFrame(rows)
        y = np.array(labels, dtype=int)
        self.feature_names = list(df_X.columns)

        print(f"Total Transactions: {len(df_X):,} | Fraud Count: {y.sum()} ({y.mean()*100:.3f}%)")
        print(f"Feature Dimensions: {len(self.feature_names)} features")

        # Stratified train/test split to preserve fraud proportion
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            df_X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df)
        X_test = scaler.transform(X_test_df)

        return X_train, X_test, y_train, y_test, scaler

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        imbalance_method: str,
    ) -> EvaluationReport:
        """Computes comprehensive imbalanced classification metrics."""
        y_pred = model.predict(X_test)
        
        # Determine prediction probabilities for PR-AUC / ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = precision_score(y_test, y_pred, labels=[0, 1], zero_division=0)
        recall = recall_score(y_test, y_pred, labels=[0, 1], zero_division=0)
        f1 = f1_score(y_test, y_pred, labels=[0, 1], zero_division=0)

        # Handle edge case where test set might contain only one class in small unit tests
        if len(np.unique(y_test)) > 1:
            pr_auc = average_precision_score(y_test, y_prob)
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            pr_auc = float(precision)
            roc_auc = 0.5

        accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))

        return EvaluationReport(
            model_name=model_name,
            imbalance_method=imbalance_method,
            precision=round(float(precision), 4),
            recall=round(float(recall), 4),
            f1=round(float(f1), 4),
            pr_auc=round(float(pr_auc), 4),
            roc_auc=round(float(roc_auc), 4),
            accuracy=round(float(accuracy), 4),
            confusion_matrix=cm.tolist(),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
        )

    def train_and_benchmark(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Tuple[Any, EvaluationReport]]:
        """
        Trains and benchmarks:
        1. Logistic Regression (Class Weighting)
        2. Logistic Regression (SMOTE)
        3. Random Forest (Class Weighting)
        4. Random Forest (SMOTE)
        5. XGBoost (scale_pos_weight)
        6. XGBoost (SMOTE)
        """
        results = {}

        # Precompute SMOTE training split
        print("\nApplying SMOTE on training split...")
        pos_samples = int(y_train.sum())
        k_neighbors = min(5, max(1, pos_samples - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"Original Train Balance: {np.bincount(y_train)} -> SMOTE Balance: {np.bincount(y_train_smote)}")

        # Calculate scale_pos_weight for XGBoost
        neg_count, pos_count = np.bincount(y_train)
        pos_weight = float(neg_count / max(1, pos_count))

        # 1. Logistic Regression (Class Weighted)
        print("\n[1/6] Training Logistic Regression (Class Weighted)...")
        lr_cw = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr_cw.fit(X_train, y_train)
        results["LogReg (Class Weight)"] = (
            lr_cw,
            self.evaluate_model(lr_cw, X_test, y_test, "Logistic Regression", "Class Weighting")
        )

        # 2. Logistic Regression (SMOTE)
        print("[2/6] Training Logistic Regression (SMOTE)...")
        lr_sm = LogisticRegression(max_iter=1000, random_state=42)
        lr_sm.fit(X_train_smote, y_train_smote)
        results["LogReg (SMOTE)"] = (
            lr_sm,
            self.evaluate_model(lr_sm, X_test, y_test, "Logistic Regression", "SMOTE")
        )

        # 3. Random Forest (Class Weighted)
        print("[3/6] Training Random Forest (Class Weighted)...")
        rf_cw = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_cw.fit(X_train, y_train)
        results["Random Forest (Class Weight)"] = (
            rf_cw,
            self.evaluate_model(rf_cw, X_test, y_test, "Random Forest", "Class Weighting")
        )

        # 4. Random Forest (SMOTE)
        print("[4/6] Training Random Forest (SMOTE)...")
        rf_sm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_sm.fit(X_train_smote, y_train_smote)
        results["Random Forest (SMOTE)"] = (
            rf_sm,
            self.evaluate_model(rf_sm, X_test, y_test, "Random Forest", "SMOTE")
        )

        # 5. XGBoost (Class Weighted: scale_pos_weight)
        print("[5/6] Training XGBoost (scale_pos_weight)...")
        xgb_cw = xgb.XGBClassifier(
            scale_pos_weight=pos_weight,
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_cw.fit(X_train, y_train)
        results["XGBoost (Class Weight)"] = (
            xgb_cw,
            self.evaluate_model(xgb_cw, X_test, y_test, "XGBoost", "scale_pos_weight")
        )

        # 6. XGBoost (SMOTE)
        print("[6/6] Training XGBoost (SMOTE)...")
        xgb_sm = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_sm.fit(X_train_smote, y_train_smote)
        results["XGBoost (SMOTE)"] = (
            xgb_sm,
            self.evaluate_model(xgb_sm, X_test, y_test, "XGBoost", "SMOTE")
        )

        return results

    def save_champion_model(
        self,
        model: Any,
        scaler: StandardScaler,
        report: EvaluationReport,
        version: str = "v1",
    ) -> Tuple[Path, Path]:
        """Saves champion model, scaler, and metadata JSON."""
        model_path = self.models_dir / f"model_{version}.pkl"
        scaler_path = self.models_dir / "scaler.pkl"
        meta_path = self.models_dir / f"metadata_{version}.json"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        metadata = {
            "version": version,
            "model_type": report.model_name,
            "imbalance_method": report.imbalance_method,
            "metrics": report.to_dict(),
            "feature_names": self.feature_names,
            "num_features": len(self.feature_names),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nSaved champion model -> {model_path}")
        print(f"Saved metadata       -> {meta_path}")
        return model_path, meta_path
