"""
Real-Time Feature Engineering Pipeline & SQLite Feature Store Integration.

Computes stateful rolling-window features per user in real time:
- Rolling average spend (1h window)
- Rolling total spend (1h window)
- Transaction frequency / velocity (last 1 hour)
- Time elapsed since last user transaction
- Spending deviation from user baseline
- Log-transformed amount and cyclical hour of day

Persists computed vectors to SQLite for sub-millisecond retrieval by transaction ID.
"""

import math
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from src.ingestion.streamer import TransactionEvent
from db.storage import FeatureStoreDB, DEFAULT_DB_PATH


@dataclass
class EnrichedFeatureVector:
    """Represents a transaction enriched with both raw and engineered streaming features."""
    transaction_id: str
    user_id: str
    timestamp: str
    simulated_seconds: float
    amount: float
    amount_log: float
    hour_of_day: int
    time_since_last_tx: float
    tx_count_1h: int
    rolling_avg_spend_1h: float
    rolling_spend_sum_1h: float
    spending_deviation: float
    raw_features: Dict[str, float]
    label: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert enriched record to dictionary."""
        data = {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "simulated_seconds": self.simulated_seconds,
            "amount": self.amount,
            "amount_log": self.amount_log,
            "hour_of_day": self.hour_of_day,
            "time_since_last_tx": self.time_since_last_tx,
            "tx_count_1h": self.tx_count_1h,
            "rolling_avg_spend_1h": self.rolling_avg_spend_1h,
            "rolling_spend_sum_1h": self.rolling_spend_sum_1h,
            "spending_deviation": self.spending_deviation,
            "label": self.label,
        }
        # Add raw features (e.g. V1..V28)
        data.update(self.raw_features)
        return data

    def get_feature_matrix_row(self) -> Dict[str, float]:
        """Numeric-only feature dictionary ready for ML model inference."""
        feats = {
            "amount": self.amount,
            "amount_log": self.amount_log,
            "hour_of_day": float(self.hour_of_day),
            "time_since_last_tx": self.time_since_last_tx,
            "tx_count_1h": float(self.tx_count_1h),
            "rolling_avg_spend_1h": self.rolling_avg_spend_1h,
            "rolling_spend_sum_1h": self.rolling_spend_sum_1h,
            "spending_deviation": self.spending_deviation,
        }
        feats.update(self.raw_features)
        return feats


class FeatureEngineer:
    """
    Stateful streaming feature engineering engine.
    Maintains active per-user transaction histories and writes to SQLite FeatureStoreDB.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        window_seconds: float = 3600.0, # 1 hour default rolling window
    ):
        self.window_seconds = window_seconds
        self.db = FeatureStoreDB(db_path=db_path)
        
        # User history state: user_id -> deque of (simulated_seconds, amount)
        self._user_history: Dict[str, deque[Tuple[float, float]]] = defaultdict(deque)
        # Most recent transaction timestamp per user
        self._last_tx_time: Dict[str, float] = {}

    def _evict_old_records(self, user_id: str, current_time: float):
        """Purge records outside the rolling time window."""
        queue = self._user_history[user_id]
        cutoff = current_time - self.window_seconds
        while queue and queue[0][0] < cutoff:
            queue.popleft()

    def process_transaction(self, event: TransactionEvent, persist: bool = True) -> EnrichedFeatureVector:
        """
        Takes a raw TransactionEvent, calculates rolling features,
        and persists the enriched feature vector into SQLite.
        """
        user_id = event.user_id
        curr_time = event.simulated_seconds
        amount = event.amount

        # 1. Time since last transaction for this user
        if user_id in self._last_tx_time:
            time_since_last = max(0.0, curr_time - self._last_tx_time[user_id])
        else:
            time_since_last = 86400.0 # Default to 24 hours for first known transaction

        # 2. Evict expired transactions from the rolling window
        self._evict_old_records(user_id, curr_time)
        history = self._user_history[user_id]

        # 3. Compute rolling statistics across prior transactions in the 1h window
        if len(history) > 0:
            amounts = [h[1] for h in history]
            tx_count_1h = len(amounts)
            rolling_spend_sum = sum(amounts)
            rolling_avg_spend = rolling_spend_sum / tx_count_1h

            # Standard deviation for deviation metric
            variance = sum((x - rolling_avg_spend) ** 2 for x in amounts) / tx_count_1h
            rolling_std = math.sqrt(variance)
            # Z-score-like deviation normalized against user history
            spending_deviation = (amount - rolling_avg_spend) / (rolling_std + 1.0)
        else:
            tx_count_1h = 0
            rolling_spend_sum = 0.0
            rolling_avg_spend = amount
            spending_deviation = 0.0

        # 4. Cyclical and amount transformations
        amount_log = math.log1p(max(0.0, amount))
        hour_of_day = int((curr_time // 3600) % 24)

        # 5. Update user state with current transaction
        history.append((curr_time, amount))
        self._last_tx_time[user_id] = curr_time

        # 6. Build enriched vector
        enriched = EnrichedFeatureVector(
            transaction_id=event.transaction_id,
            user_id=user_id,
            timestamp=event.timestamp,
            simulated_seconds=curr_time,
            amount=amount,
            amount_log=amount_log,
            hour_of_day=hour_of_day,
            time_since_last_tx=round(time_since_last, 2),
            tx_count_1h=tx_count_1h,
            rolling_avg_spend_1h=round(rolling_avg_spend, 2),
            rolling_spend_sum_1h=round(rolling_spend_sum, 2),
            spending_deviation=round(spending_deviation, 4),
            raw_features=event.features,
            label=event.label,
        )

        # 7. Persist to SQLite feature store
        if persist:
            record_dict = enriched.to_dict()
            record_dict["raw_features"] = event.features
            self.db.save_feature_vector(record_dict)

        return enriched

    def get_feature_vector(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored feature vector by transaction ID from SQLite."""
        return self.db.get_by_transaction_id(transaction_id)
