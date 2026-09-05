"""
Real-time Transaction Ingestion & Streaming Simulation Module.

Simulates real-time transaction streaming row-by-row without loading
the entire dataset into memory. Designed with a generic schema adapter
so datasets like Kaggle CreditCard or IEEE-CIS can be ingested interchangeably.
"""

import csv
import time
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Generator, Optional, List, Callable


@dataclass
class TransactionEvent:
    """Standardized transaction event object produced by the stream."""
    transaction_id: str
    timestamp: str               # ISO-8601 simulated wall-clock or relative timestamp
    simulated_seconds: float     # Dataset 'Time' field (seconds elapsed)
    user_id: str                 # Real or pseudo user identifier
    amount: float                # Transaction currency amount
    features: Dict[str, float]   # PCA / raw feature dictionary (e.g. V1..V28)
    label: Optional[int] = None  # Ground truth if available (0 = legit, 1 = fraud)
    raw_record: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten event into a single dictionary representation."""
        res = {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
            "simulated_seconds": self.simulated_seconds,
            "user_id": self.user_id,
            "amount": self.amount,
            "label": self.label,
        }
        res.update(self.features)
        return res

    def summary(self) -> str:
        """Formatted single-line summary for logging."""
        flag_str = "[FRAUD ALERT]" if self.label == 1 else "[LEGIT]"
        return (
            f"{self.timestamp} | ID: {self.transaction_id:<10} | User: {self.user_id:<9} | "
            f"Amt: ${self.amount:>8.2f} | Status: {flag_str}"
        )


@dataclass
class StreamSchemaConfig:
    """Schema configuration allowing seamless dataset swapping (e.g., Kaggle to IEEE-CIS)."""
    time_col: str = "Time"
    amount_col: str = "Amount"
    label_col: Optional[str] = "Class"
    user_id_col: Optional[str] = None       # If None, pseudo-user will be assigned
    num_pseudo_users: int = 500             # Cardinality of simulated user base
    exclude_feature_cols: List[str] = field(
        default_factory=lambda: ["Time", "Amount", "Class", "transaction_id", "user_id"]
    )


class TransactionStreamer:
    """
    Simulates real-time transaction event streaming row-by-row from CSV.
    Uses generator execution to ensure O(1) memory footprint.
    """

    def __init__(
        self,
        csv_path: str | Path,
        delay_seconds: float = 0.05,
        schema: Optional[StreamSchemaConfig] = None,
        base_timestamp: Optional[datetime] = None,
    ):
        self.csv_path = Path(csv_path)
        self.delay_seconds = delay_seconds
        self.schema = schema or StreamSchemaConfig()
        self.base_timestamp = base_timestamp or datetime.now(timezone.utc)

        # Stream statistics
        self.total_streamed = 0
        self.fraud_streamed = 0
        self.start_wall_time: Optional[float] = None

    def _assign_pseudo_user(self, row_idx: int, amount: float, time_sec: float) -> str:
        """
        Derives a deterministic pseudo user ID for anonymized datasets.
        Combines cyclical time hashing and row index to create realistic repeat user histories.
        """
        user_num = (int(time_sec // 60) + (row_idx % 23) * 17) % self.schema.num_pseudo_users
        return f"usr_{user_num:04d}"

    def stream(
        self,
        max_events: Optional[int] = None,
        start_row: int = 0,
    ) -> Generator[TransactionEvent, None, None]:
        """
        Yields transactions one by one with a simulated delay.
        Never loads the full file into memory.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at: {self.csv_path.resolve()}")

        self.total_streamed = 0
        self.fraud_streamed = 0
        self.start_wall_time = time.time()

        with open(self.csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row_idx, row in enumerate(reader):
                if row_idx < start_row:
                    continue
                if max_events is not None and self.total_streamed >= max_events:
                    break

                # Extract time and amount
                simulated_sec = float(row.get(self.schema.time_col, row_idx))
                amount = float(row.get(self.schema.amount_col, 0.0))

                # Compute simulated event timestamp
                event_time = self.base_timestamp + timedelta(seconds=simulated_sec)
                timestamp_str = event_time.strftime("%Y-%m-%d %H:%M:%S")

                # Extract label
                label = None
                if self.schema.label_col and self.schema.label_col in row:
                    try:
                        label = int(float(row[self.schema.label_col]))
                    except (ValueError, TypeError):
                        label = 0

                # Determine user ID
                if self.schema.user_id_col and self.schema.user_id_col in row:
                    user_id = str(row[self.schema.user_id_col])
                else:
                    user_id = self._assign_pseudo_user(row_idx, amount, simulated_sec)

                # Extract features (all columns except excluded ones)
                features = {}
                for col, val in row.items():
                    if col not in self.schema.exclude_feature_cols:
                        try:
                            features[col] = float(val)
                        except (ValueError, TypeError):
                            pass

                event = TransactionEvent(
                    transaction_id=f"tx_{row_idx + 1:07d}",
                    timestamp=timestamp_str,
                    simulated_seconds=simulated_sec,
                    user_id=user_id,
                    amount=amount,
                    features=features,
                    label=label,
                    raw_record=row,
                )

                self.total_streamed += 1
                if label == 1:
                    self.fraud_streamed += 1

                # Apply inter-transaction delay if configured
                if self.delay_seconds > 0:
                    time.sleep(self.delay_seconds)

                yield event


class QueueStreamer:
    """
    Decoupled streaming engine running in a separate background thread
    pushing events into a thread-safe queue.Queue. Downstream consumers
    (e.g., feature extraction, scoring) can read events asynchronously.
    """

    def __init__(
        self,
        csv_path: str | Path,
        delay_seconds: float = 0.05,
        maxsize: int = 1000,
        schema: Optional[StreamSchemaConfig] = None,
    ):
        self.streamer = TransactionStreamer(csv_path, delay_seconds, schema)
        self.queue: queue.Queue[Optional[TransactionEvent]] = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _producer(self, max_events: Optional[int]):
        """Internal background worker thread populating queue."""
        try:
            for event in self.streamer.stream(max_events=max_events):
                if self._stop_event.is_set():
                    break
                self.queue.put(event, block=True, timeout=5.0)
        except Exception as e:
            print(f"[QueueStreamer] Producer encountered exception: {e}")
        finally:
            # Sentinel None indicates stream completion
            self.queue.put(None)

    def start(self, max_events: Optional[int] = None):
        """Start streaming producer in background."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._producer,
            args=(max_events,),
            daemon=True,
            name="StreamerProducerThread"
        )
        self._thread.start()

    def stop(self):
        """Signal producer thread to stop and clean up."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get(self, timeout: Optional[float] = 1.0) -> Optional[TransactionEvent]:
        """Fetch next transaction event from queue. Returns None when stream is depleted."""
        try:
            return self.queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
