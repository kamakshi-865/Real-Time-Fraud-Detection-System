"""
SQLite Database Layer for Feature Store and Case Management.
Manages persistent storage for feature vectors, transactions, and reviewer feedback.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List, Generator

DB_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = DB_DIR / "fraud_detection.db"


class FeatureStoreDB:
    """Manages SQLite connection and operations for the feature store."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing a safe auto-closing SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initializes tables and indices for feature store."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_store (
                transaction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                simulated_seconds REAL NOT NULL,
                amount REAL NOT NULL,
                amount_log REAL NOT NULL,
                hour_of_day INTEGER NOT NULL,
                time_since_last_tx REAL NOT NULL,
                tx_count_1h INTEGER NOT NULL,
                rolling_avg_spend_1h REAL NOT NULL,
                rolling_spend_sum_1h REAL NOT NULL,
                spending_deviation REAL NOT NULL,
                features_json TEXT NOT NULL,
                label INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_user_time 
            ON feature_store(user_id, simulated_seconds);
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_store_label 
            ON feature_store(label);
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS flagged_cases (
                case_id TEXT PRIMARY KEY,
                transaction_id TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                risk_score REAL NOT NULL,
                status TEXT DEFAULT 'PENDING',
                shap_summary TEXT,
                shap_features_json TEXT,
                reviewer_decision TEXT,
                reviewer_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TIMESTAMP
            );
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviewer_decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                amount REAL NOT NULL,
                risk_score REAL NOT NULL,
                decision TEXT NOT NULL,
                notes TEXT,
                reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            conn.commit()

    def save_feature_vector(self, record: Dict[str, Any]) -> None:
        """Insert or replace a computed feature vector into SQLite."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO feature_store (
                transaction_id, user_id, timestamp, simulated_seconds,
                amount, amount_log, hour_of_day, time_since_last_tx,
                tx_count_1h, rolling_avg_spend_1h, rolling_spend_sum_1h,
                spending_deviation, features_json, label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record["transaction_id"],
                record["user_id"],
                record["timestamp"],
                record["simulated_seconds"],
                record["amount"],
                record["amount_log"],
                record["hour_of_day"],
                record["time_since_last_tx"],
                record["tx_count_1h"],
                record["rolling_avg_spend_1h"],
                record["rolling_spend_sum_1h"],
                record["spending_deviation"],
                json.dumps(record.get("raw_features", {})),
                record.get("label")
            ))
            conn.commit()

    def get_by_transaction_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored feature vector by its transaction ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feature_store WHERE transaction_id = ?", (transaction_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            if "features_json" in res and res["features_json"]:
                res["raw_features"] = json.loads(res["features_json"])
            return res

    def get_recent_features(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch the most recently stored feature vectors."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM feature_store 
            ORDER BY simulated_seconds DESC 
            LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if "features_json" in item and item["features_json"]:
                    item["raw_features"] = json.loads(item["features_json"])
                results.append(item)
            return results

    def get_total_count(self) -> int:
        """Returns total records stored in the feature store."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feature_store")
            return cursor.fetchone()[0]

    def clear(self) -> None:
        """Purge all rows from feature_store (useful for testing/resetting)."""
        with self.get_connection() as conn:
            conn.cursor().execute("DELETE FROM feature_store")
            conn.cursor().execute("DELETE FROM flagged_cases")
            conn.cursor().execute("DELETE FROM reviewer_decisions")
            conn.commit()

    # --- Case Management & Reviewer Feedback Methods ---

    def save_flagged_case(self, case_data: Dict[str, Any]) -> None:
        """Saves or updates a suspicious transaction flagged for reviewer triage."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO flagged_cases (
                case_id, transaction_id, user_id, timestamp, amount,
                risk_score, status, shap_summary, shap_features_json,
                reviewer_decision, reviewer_notes, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case_data.get("case_id", f"case_{case_data['transaction_id']}"),
                case_data["transaction_id"],
                case_data["user_id"],
                case_data["timestamp"],
                case_data["amount"],
                case_data["risk_score"],
                case_data.get("status", "PENDING"),
                case_data.get("shap_summary", ""),
                json.dumps(case_data.get("shap_features", [])),
                case_data.get("reviewer_decision"),
                case_data.get("reviewer_notes"),
                case_data.get("reviewed_at")
            ))
            conn.commit()

    def get_flagged_cases(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves flagged cases, optionally filtered by status ('PENDING', 'CONFIRMED_FRAUD', 'FALSE_ALARM')."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status_filter and status_filter.upper() != "ALL":
                cursor.execute("""
                SELECT * FROM flagged_cases 
                WHERE status = ? 
                ORDER BY created_at DESC
                """, (status_filter.upper(),))
            else:
                cursor.execute("""
                SELECT * FROM flagged_cases 
                ORDER BY created_at DESC
                """)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("shap_features_json"):
                    try:
                        item["shap_features"] = json.loads(item["shap_features_json"])
                    except json.JSONDecodeError:
                        item["shap_features"] = []
                results.append(item)
            return results

    def record_reviewer_decision(
        self,
        transaction_id: str,
        decision: str, # 'CONFIRMED_FRAUD' or 'FALSE_ALARM'
        notes: str = ""
    ) -> None:
        """
        Records human reviewer action:
        1. Logs into reviewer_decisions table for model feedback loop (Module 8).
        2. Updates flagged_cases status from 'PENDING' to decision.
        """
        import datetime
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        decision_norm = decision.upper()

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Fetch case details
            cursor.execute("SELECT user_id, amount, risk_score FROM flagged_cases WHERE transaction_id = ?", (transaction_id,))
            case_row = cursor.fetchone()
            if case_row:
                u_id, amt, r_score = case_row[0], case_row[1], case_row[2]
            else:
                u_id, amt, r_score = "usr_unknown", 0.0, 1.0

            # 2. Insert into reviewer_decisions audit table
            cursor.execute("""
            INSERT INTO reviewer_decisions (
                transaction_id, user_id, amount, risk_score, decision, notes, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (transaction_id, u_id, amt, r_score, decision_norm, notes, now_str))

            # 3. Update case status in flagged_cases
            cursor.execute("""
            UPDATE flagged_cases 
            SET status = ?, reviewer_decision = ?, reviewer_notes = ?, reviewed_at = ?
            WHERE transaction_id = ?
            """, (decision_norm, decision_norm, notes, now_str, transaction_id))

            conn.commit()

    def get_reviewer_stats(self) -> Dict[str, int]:
        """Summary counts of pending vs resolved fraud cases."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM flagged_cases WHERE status = 'PENDING'")
            pending = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM flagged_cases WHERE status = 'CONFIRMED_FRAUD'")
            confirmed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM flagged_cases WHERE status = 'FALSE_ALARM'")
            false_alarms = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM flagged_cases")
            total = cursor.fetchone()[0]

            return {
                "total_flagged": total,
                "pending": pending,
                "confirmed_fraud": confirmed,
                "false_alarms": false_alarms,
            }

    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch chronological log of reviewer audit decisions."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM reviewer_decisions 
            ORDER BY reviewed_at DESC 
            LIMIT ?
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

