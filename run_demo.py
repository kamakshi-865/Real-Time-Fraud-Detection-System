"""
Unified CLI Entry Point for Real-Time Fraud Detection System Demos.

Usage:
  python run_demo.py --module 1                 # Run Module 1 streaming demo
  python run_demo.py --module 1 --mode queue    # Run Module 1 queue producer-consumer demo
  python run_demo.py --module 1 --limit 30 --delay 0.05
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 encoding support for Windows PowerShell console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.ingestion.streamer import TransactionStreamer, QueueStreamer, StreamSchemaConfig, TransactionEvent
from data.setup_data import DEFAULT_CSV_PATH, check_dataset_status, generate_synthetic_dataset
from src.models.train import MODELS_DIR


# ANSI terminal styling for rich presentation
class Style:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def run_module_1_demo(max_events: int = 25, delay: float = 0.08, mode: str = "generator"):
    """Demonstrate Module 1: Row-by-row streaming data ingestion."""
    print(f"\n{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}  MODULE 1 DEMO: REAL-TIME TRANSACTION INGESTION & STREAMING REPLAY   {Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}\n")

    # Verify dataset exists
    status = check_dataset_status(DEFAULT_CSV_PATH)
    if not status["exists"]:
        print(f"{Style.YELLOW}[!] Dataset missing. Auto-generating sample dataset for demo...{Style.RESET}")
        generate_synthetic_dataset(DEFAULT_CSV_PATH, num_rows=2000, fraud_rate=0.04)
        status = check_dataset_status(DEFAULT_CSV_PATH)

    print(f"{Style.BOLD}Dataset Path:{Style.RESET}       {status['path']}")
    print(f"{Style.BOLD}Total In Dataset:{Style.RESET}   {status['rows']:,} records ({status['fraud_count']} fraud / {status['fraud_ratio']*100:.2f}%)")
    print(f"{Style.BOLD}Stream Mode:{Style.RESET}        {mode.upper()} ({'Thread-safe Queue' if mode == 'queue' else 'Python Generator'})")
    print(f"{Style.BOLD}Simulated Delay:{Style.RESET}    {delay}s per transaction")
    print(f"{Style.BOLD}Display Limit:{Style.RESET}      {max_events} events")
    print(f"\n{Style.DIM}{'-' * 88}{Style.RESET}")
    print(
        f"{Style.BOLD}{'EVENT TIMESTAMP':<20} | {'TX ID':<10} | {'USER':<9} | {'AMOUNT':>9} | {'STATUS':<14} | {'SAMPLE FEATURES (V1, V2, V3)'}{Style.RESET}"
    )
    print(f"{Style.DIM}{'-' * 88}{Style.RESET}")

    start_time = time.time()
    count = 0
    fraud_count = 0

    if mode == "queue":
        # Demonstrate asynchronous queue-based consumer
        q_streamer = QueueStreamer(DEFAULT_CSV_PATH, delay_seconds=delay)
        q_streamer.start(max_events=max_events)
        try:
            while True:
                event = q_streamer.get(timeout=2.0)
                if event is None:
                    break
                count += 1
                if event.label == 1:
                    fraud_count += 1
                _display_event(event)
        finally:
            q_streamer.stop()
    else:
        # Standard generator stream
        streamer = TransactionStreamer(DEFAULT_CSV_PATH, delay_seconds=delay)
        for event in streamer.stream(max_events=max_events):
            count += 1
            if event.label == 1:
                fraud_count += 1
            _display_event(event)

    elapsed = time.time() - start_time
    rate = count / elapsed if elapsed > 0 else 0.0

    print(f"{Style.DIM}{'-' * 88}{Style.RESET}\n")
    print(f"{Style.BOLD}{Style.GREEN}[OK] Ingestion Stream Completed Successfully!{Style.RESET}")
    print(f"  * Processed Events:    {Style.BOLD}{count}{Style.RESET}")
    print(f"  * Flagged Fraud:       {Style.BOLD}{fraud_count}{Style.RESET}")
    print(f"  * Wall Time Elapsed:   {elapsed:.2f} seconds")
    print(f"  * Effective Rate:      {rate:.1f} transactions/second")
    print(f"  * Memory Footprint:    O(1) Streaming (Zero bulk file load into RAM)")
    print(f"\n{Style.DIM}Ready to proceed to Module 2 (Feature Engineering & SQLite Feature Store).{Style.RESET}\n")


def _display_event(event):
    """Formats and prints an arriving transaction event."""
    if event.label == 1:
        status_tag = f"{Style.RED}{Style.BOLD}[FRAUD ALERT] {Style.RESET}"
        amt_str = f"{Style.RED}${event.amount:>8.2f}{Style.RESET}"
    else:
        status_tag = f"{Style.GREEN}[LEGIT]       {Style.RESET}"
        amt_str = f"${event.amount:>8.2f}"

    v1 = event.features.get("V1", 0.0)
    v2 = event.features.get("V2", 0.0)
    v3 = event.features.get("V3", 0.0)
    feat_preview = f"V1={v1:+.2f}, V2={v2:+.2f}, V3={v3:+.2f}"

    print(
        f"{event.timestamp:<20} | {event.transaction_id:<10} | {event.user_id:<9} | {amt_str} | {status_tag} | {Style.DIM}{feat_preview}{Style.RESET}"
    )

def run_module_2_demo(tx_limit: int = 5):
    """Demonstrate Module 2: Feature Engineering & SQLite Feature Store."""
    from src.features.pipeline import FeatureEngineer
    from db.storage import DEFAULT_DB_PATH

    print(f"\n{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}  MODULE 2 DEMO: FEATURE ENGINEERING & SQLITE FEATURE STORE            {Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}\n")

    engineer = FeatureEngineer(db_path=DEFAULT_DB_PATH, window_seconds=3600.0)
    print(f"{Style.BOLD}SQLite Database:{Style.RESET}    {DEFAULT_DB_PATH.resolve()}")
    print(f"{Style.BOLD}Rolling Window:{Style.RESET}     1 Hour (3,600 seconds)")
    print(f"{Style.BOLD}Initial Store Count:{Style.RESET}{engineer.db.get_total_count():,} records\n")

    # Step 1: Stream transactions and enrich them in real-time
    streamer = TransactionStreamer(DEFAULT_CSV_PATH, delay_seconds=0.0)
    events = list(streamer.stream(max_events=tx_limit))
    
    print(f"{Style.BOLD}--- Step 1: Ingesting Raw Transactions & Computing Features ---{Style.RESET}")
    enriched_list = []
    for evt in events:
        enriched = engineer.process_transaction(evt, persist=True)
        enriched_list.append(enriched)
        print(f"\n  [RAW TRANSACTION INGESTED]")
        print(f"  • ID: {evt.transaction_id} | User: {evt.user_id} | Amount: ${evt.amount:.2f} | Time: {evt.timestamp}")
        print(f"  • PCA Features: V1={evt.features.get('V1', 0.0):+.3f}, V2={evt.features.get('V2', 0.0):+.3f}, ... (28 total)")
        
        print(f"  {Style.GREEN}--> [ENGINEERED FEATURES COMPUTED]{Style.RESET}")
        print(f"      - Time Since Last Tx:    {enriched.time_since_last_tx:>9.1f} s")
        print(f"      - 1h Transaction Count:  {enriched.tx_count_1h:>9d}")
        print(f"      - 1h Rolling Avg Spend:  ${enriched.rolling_avg_spend_1h:>8.2f}")
        print(f"      - 1h Rolling Total Spend:${enriched.rolling_spend_sum_1h:>8.2f}")
        print(f"      - Spending Deviation:    {enriched.spending_deviation:>9.4f} (std units)")
        print(f"      - Log Normalized Amount: {enriched.amount_log:>9.4f}")
        print(f"      - Cyclical Hour of Day:  {enriched.hour_of_day:>9d}:00")

    # Step 2: Demonstrate targeted retrieval from SQLite Feature Store
    sample_target = enriched_list[0].transaction_id
    print(f"\n{Style.BOLD}--- Step 2: Querying Feature Store by Transaction ID ---{Style.RESET}")
    print(f"Query: {Style.CYAN}SELECT * FROM feature_store WHERE transaction_id = '{sample_target}'{Style.RESET}")
    
    record = engineer.get_feature_vector(sample_target)
    if record:
        print(f"\n{Style.GREEN}[OK] Successfully retrieved record from SQLite!{Style.RESET}")
        print(f"  • Transaction ID:     {record['transaction_id']}")
        print(f"  • User ID:            {record['user_id']}")
        print(f"  • Timestamp:          {record['timestamp']}")
        print(f"  • Amount:             ${record['amount']:.2f}")
        print(f"  • Time Since Last Tx: {record['time_since_last_tx']}s")
        print(f"  • 1h Tx Velocity:     {record['tx_count_1h']} prior txs")
        print(f"  • 1h Rolling Avg:     ${record['rolling_avg_spend_1h']:.2f}")
        print(f"  • Stored JSON Length: {len(record['features_json'])} chars")
        print(f"  • Record Created At:  {record['created_at']}")
    else:
        print(f"{Style.RED}[ERROR] Record not found in store!{Style.RESET}")

    # Step 3: Demonstrate repeat user velocity behavior
    print(f"\n{Style.BOLD}--- Step 3: Demonstrating Behavioral Velocity on Repeat User ---{Style.RESET}")
    test_user = "usr_demo_vip"
    print(f"Simulating 3 transactions for user '{test_user}' within 15 minutes:")

    base_sec = 5000.0
    amounts = [45.0, 55.0, 850.0] # Normal, Normal, Sudden High Spike
    delays = [0.0, 180.0, 600.0]   # 0s, 3m later, 10m later

    for idx, (amt, delta) in enumerate(zip(amounts, delays), start=1):
        test_evt = TransactionEvent(
            transaction_id=f"tx_demo_user_{idx:03d}",
            timestamp=f"2026-09-06 01:{idx*5:02d}:00",
            simulated_seconds=base_sec + delta,
            user_id=test_user,
            amount=amt,
            features={"V1": 0.25, "V2": -0.85},
            label=1 if amt > 500 else 0
        )
        res = engineer.process_transaction(test_evt, persist=True)
        flag = f"{Style.RED}[ANOMALY SPIKE]{Style.RESET}" if res.spending_deviation > 2.0 else "[NORMAL]"
        print(
            f"  Tx #{idx} (${amt:>6.2f} @ +{delta:>4.0f}s) -> "
            f"Prior Txs (1h): {res.tx_count_1h} | "
            f"Rolling Avg: ${res.rolling_avg_spend_1h:>6.2f} | "
            f"Deviation: {res.spending_deviation:>+6.2f} {flag}"
        )

    print(f"\n{Style.DIM}{'-' * 88}{Style.RESET}")
    print(f"{Style.BOLD}{Style.GREEN}[OK] Module 2 Demo Completed Successfully!{Style.RESET}")
    print(f"  * SQLite Store Total Records: {Style.BOLD}{engineer.db.get_total_count():,}{Style.RESET}")
    print(f"  * Feature Vector Dimensions:  {len(enriched_list[0].get_feature_matrix_row())} total model-ready features")
    print(f"\n{Style.DIM}Ready to proceed to Module 3 (Model Training: Baseline & XGBoost with SMOTE).{Style.RESET}\n")


def run_module_3_demo(max_rows: int = 5000):
    """Demonstrate Module 3: Baseline & XGBoost Training with SMOTE vs Class Weighting."""
    from src.models.train import ModelTrainer, MODELS_DIR

    print(f"\n{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}  MODULE 3 DEMO: MODEL TRAINING & CLASS IMBALANCE BENCHMARKING          {Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}\n")

    # Step 1: Analytical explanation of why accuracy is deceptive in fraud detection
    print(f"{Style.BOLD}--- Why Accuracy Is a Deceptive Metric in Fraud Detection ---{Style.RESET}")
    print(
        f"In transaction fraud datasets, fraudulent cases typically represent {Style.BOLD}~0.17% to 0.5%{Style.RESET} of records.\n"
        f"• A naive dummy classifier that simply flags every single transaction as {Style.GREEN}'Legitimate'{Style.RESET}\n"
        f"  achieves a seemingly stellar {Style.BOLD}{Style.RED}99.83% Accuracy{Style.RESET}, yet catches {Style.BOLD}0 frauds{Style.RESET} (Recall = 0.0).\n"
        f"• To build a viable financial defence system, we evaluate with:\n"
        f"    1. {Style.BOLD}PR-AUC (Precision-Recall AUC){Style.RESET}: Primary measure of detection power under extreme skew.\n"
        f"    2. {Style.BOLD}Recall{Style.RESET}: Percentage of real fraud prevented.\n"
        f"    3. {Style.BOLD}Precision{Style.RESET}: Minimizing false positive customer friction.\n"
        f"    4. {Style.BOLD}F1 Score{Style.RESET}: Harmonic mean balancing Precision and Recall.\n"
    )

    trainer = ModelTrainer(DEFAULT_CSV_PATH, models_dir=MODELS_DIR)
    
    # Step 2: Prepare dataset and stratify
    print(f"{Style.BOLD}--- Step 2: Preparing Features & Stratified Split ---{Style.RESET}")
    X_train, X_test, y_train, y_test, scaler = trainer.prepare_dataset(max_rows=max_rows, test_size=0.2)

    # Step 3: Run full benchmark across models and imbalance handlers
    print(f"\n{Style.BOLD}--- Step 3: Training & Comparing All Models & Imbalance Techniques ---{Style.RESET}")
    results = trainer.train_and_benchmark(X_train, X_test, y_train, y_test)

    # Step 4: Render performance comparison table
    print(f"\n{Style.BOLD}======================================================================================{Style.RESET}")
    print(f"{Style.BOLD}{'MODEL':<18} | {'IMBALANCE METHOD':<18} | {'PR-AUC':>7} | {'RECALL':>7} | {'PREC':>7} | {'F1':>7} | {'ACC':>7}{Style.RESET}")
    print(f"{Style.DIM}{'-' * 86}{Style.RESET}")

    best_name = None
    best_pr_auc = -1.0
    best_model_obj = None
    best_report = None

    for label, (model_obj, rep) in results.items():
        if rep.pr_auc > best_pr_auc or (rep.pr_auc == best_pr_auc and "XGBoost" in label):
            best_pr_auc = rep.pr_auc
            best_name = label
            best_model_obj = model_obj
            best_report = rep

        print(
            f"{rep.model_name:<18} | {rep.imbalance_method:<18} | {rep.pr_auc:>7.4f} | {rep.recall:>7.4f} | "
            f"{rep.precision:>7.4f} | {rep.f1:>7.4f} | {rep.accuracy:>7.4f}"
        )

    print(f"{Style.DIM}{'-' * 86}{Style.RESET}")
    print(f"\n{Style.BOLD}{Style.GREEN}★ Champion Model Selected:{Style.RESET} {Style.BOLD}{best_name}{Style.RESET} (PR-AUC: {best_pr_auc:.4f})")

    # Step 5: Confusion Matrix for the champion model
    print(f"\n{Style.BOLD}--- Champion Confusion Matrix (Test Set: {len(y_test)} transactions) ---{Style.RESET}")
    print(f"                     [Predicted LEGIT]   [Predicted FRAUD]")
    print(f"  [Actual LEGIT]:    {best_report.tn:>15d}   {best_report.fp:>16d}  (False Alarms)")
    print(f"  [Actual FRAUD]:    {best_report.fn:>15d}   {best_report.tp:>16d}  (Caught Frauds)")

    # Step 6: Save champion model and metadata
    trainer.save_champion_model(best_model_obj, scaler, best_report, version="v1")

    print(f"\n{Style.DIM}{'-' * 86}{Style.RESET}")
    print(f"{Style.BOLD}{Style.GREEN}[OK] Module 3 Demo Completed Successfully!{Style.RESET}")
    print(f"  * Serialized Artifact: {MODELS_DIR / 'model_v1.pkl'}")
    print(f"  * Metadata Registry:   {MODELS_DIR / 'metadata_v1.json'}")
    print(f"\n{Style.DIM}Ready to proceed to Module 4 (Retraining Pipeline & Model Versioning).{Style.RESET}\n")


def run_module_4_demo():
    """Demonstrate Module 4: Retraining Pipeline & Model Versioning."""
    import json
    from src.models.retrain import RetrainingPipeline, ModelRegistry, MODELS_DIR

    print(f"\n{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}  MODULE 4 DEMO: RETRAINING PIPELINE & AUTOMATED MODEL VERSIONING       {Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}\n")

    registry = ModelRegistry(models_dir=MODELS_DIR)
    pipeline = RetrainingPipeline(csv_path=DEFAULT_CSV_PATH, models_dir=MODELS_DIR)

    # Step 1: Inspect current registry status
    current_versions = registry.get_all_versions()
    print(f"{Style.BOLD}--- Step 1: Current Model Registry State ---{Style.RESET}")
    print(f"Models Directory:   {MODELS_DIR.resolve()}")
    print(f"Existing Versions:  {current_versions if current_versions else 'None'}")
    next_ver = registry.get_next_version()
    print(f"Next Version ID:    {Style.BOLD}{Style.YELLOW}{next_ver}{Style.RESET}")

    # Step 2: Simulate new monthly data arrival and manually trigger retraining
    print(f"\n{Style.BOLD}--- Step 2: Simulating New Data Arrival & Triggering Retrain ---{Style.RESET}")
    print(f"Scenario: 'New monthly batch of transaction records arrived from payment gateway'")
    print(f"Action:   Triggering retrain for candidate {next_ver}...")

    retrain_res = pipeline.trigger_retrain(
        max_rows=5000,
        reason="monthly_scheduled_batch_refresh",
        target_version=next_ver,
    )

    # Step 3: Verify newly created versioned files on disk
    print(f"\n{Style.BOLD}--- Step 3: Verifying Serialized Artifacts on Disk ---{Style.RESET}")
    model_file = Path(retrain_res["model_path"])
    meta_file = Path(retrain_res["metadata_path"])

    print(f"  • Model Binary:    {Style.GREEN}{model_file.name}{Style.RESET} ({model_file.stat().st_size:,} bytes)")
    print(f"  • Metadata Record: {Style.GREEN}{meta_file.name}{Style.RESET} ({meta_file.stat().st_size:,} bytes)")

    # Step 4: Display the newly generated metadata JSON
    print(f"\n{Style.BOLD}--- Step 4: Inspecting Model Metadata ({meta_file.name}) ---{Style.RESET}")
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_content = json.load(f)

    print(f"  • Version:         {meta_content['version']} (Parent: {meta_content['parent_version']})")
    print(f"  • Trained At:      {meta_content['trained_at']}")
    print(f"  • Trigger Reason:  {meta_content['trigger_reason']}")
    print(f"  • Dataset Size:    {meta_content['data_summary']['total_samples']:,} transactions ({meta_content['data_summary']['fraud_count_total']} fraud)")
    print(f"  • Algorithm:       {meta_content['hyperparameters']['algorithm']} ({meta_content['hyperparameters']['imbalance_handling']})")
    print(f"  • Evaluation Metrics:")
    metrics = meta_content['metrics']
    print(f"      - PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"      - F1 Score:    {metrics['f1']:.4f}")
    print(f"      - Recall:      {metrics['recall']:.4f}")
    print(f"      - Precision:   {metrics['precision']:.4f}")
    print(f"      - Accuracy:    {metrics['accuracy']:.4f}")

    # Step 5: Updated registry manifest
    print(f"\n{Style.BOLD}--- Step 5: Updated Registry Manifest (registry.json) ---{Style.RESET}")
    all_registered = registry.get_all_versions()
    print(f"Active Version in Production: {Style.BOLD}{Style.GREEN}{meta_content['version']}{Style.RESET}")
    print(f"All Registered Versions:      {all_registered}")

    print(f"\n{Style.DIM}{'-' * 86}{Style.RESET}")
    print(f"{Style.BOLD}{Style.GREEN}[OK] Module 4 Demo Completed Successfully!{Style.RESET}")
    print(f"  * New Version Online:  {meta_content['version']}")
    print(f"  * Registry Manifest:   {MODELS_DIR / 'registry.json'}")
    print(f"\n{Style.DIM}Ready to proceed to Module 5 (Drift Detection: KS-Test & Population Stability Index).{Style.RESET}\n")


def run_module_5_demo(batch_size: int = 500):
    """Demonstrate Module 5: Statistical Drift Detection (KS-Test & PSI)."""
    import pandas as pd
    from src.monitoring.drift import DriftMonitor, create_synthetic_drifted_batch
    from src.models.retrain import ModelRegistry
    from src.features.pipeline import FeatureEngineer

    print(f"\n{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}  MODULE 5 DEMO: STATISTICAL DRIFT DETECTION (KS-TEST & PSI)            {Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}\n")

    # Step 1: Prepare baseline reference distribution
    print(f"{Style.BOLD}--- Step 1: Loading Baseline Reference Training Distribution ---{Style.RESET}")
    streamer = TransactionStreamer(DEFAULT_CSV_PATH, delay_seconds=0.0)
    engineer = FeatureEngineer(window_seconds=3600.0)
    
    ref_rows = []
    for evt in streamer.stream(max_events=2000):
        enr = engineer.process_transaction(evt, persist=False)
        ref_rows.append(enr.get_feature_matrix_row())

    df_ref = pd.DataFrame(ref_rows)
    # Monitor core behavioral and transaction features (excluding monotonic clock indicators)
    features_to_monitor = [c for c in df_ref.columns if c not in ["hour_of_day", "time_since_last_tx"]]
    print(f"Reference Window:   {len(df_ref):,} baseline transactions")
    print(f"Monitored Features: {len(features_to_monitor)} behavioral features (Amount, Velocity, Deviation, V1..V28)")
    print(f"Statistical Thresholds:")
    print(f"  • Population Stability Index (PSI): Warning >= 0.10 | Critical Alert >= 0.20")
    print(f"  • Kolmogorov-Smirnov (KS-test):     p-value < 0.01 indicates significant drift\n")

    monitor = DriftMonitor(df_ref, psi_warning_threshold=0.10, psi_alert_threshold=0.20)

    # Step 2: Test normal incoming stream batch (In-Distribution Control)
    print(f"{Style.BOLD}--- Step 2: Evaluating Normal In-Distribution Production Batch ---{Style.RESET}")
    print(f"Scenario: 'Standard daily production volume arriving from known user base'")
    normal_rows = []
    for evt in streamer.stream(max_events=batch_size, start_row=2000):
        enr = engineer.process_transaction(evt, persist=False)
        normal_rows.append(enr.get_feature_matrix_row())
    df_normal = pd.DataFrame(normal_rows)

    rep_normal = monitor.evaluate_batch(df_normal, features_to_monitor=features_to_monitor)
    print(f"Batch Size:         {rep_normal.batch_size} transactions")
    print(f"Overall Status:     {Style.BOLD}{Style.GREEN}{rep_normal.overall_status}{Style.RESET}")
    print(f"Drifted Features:   {rep_normal.drifted_features_count} / {rep_normal.total_features_monitored}")
    amt_normal = rep_normal.feature_results["amount"]
    print(
        f"Sample Check [amount]: Mean={amt_normal.current_mean:.2f} (Ref: {amt_normal.reference_mean:.2f}) | "
        f"PSI={amt_normal.psi:.4f} | KS p-val={amt_normal.ks_p_value:.4f} -> [{amt_normal.status}]"
    )

    # Step 3: Test intentionally shifted batch (Bot Attack & Macro Spend Shift)
    print(f"\n{Style.BOLD}--- Step 3: Feeding Intentionally Shifted / Drifted Batch ---{Style.RESET}")
    print(f"Scenario: 'Coordinated automated attack surge: Average amounts spike 3.5x, V4 and V12 shifted by +2.5 std'")

    df_drifted = create_synthetic_drifted_batch(
        df_normal,
        shift_features=["amount", "rolling_avg_spend_1h", "V4", "V11", "V12", "spending_deviation"],
        amount_multiplier=3.5,
        noise_shift=2.5,
    )

    rep_drifted = monitor.evaluate_batch(df_drifted, features_to_monitor=features_to_monitor)

    # Step 4: Display Alert and Itemized Drift Breakdown Table
    print(f"\n{Style.BOLD}{Style.RED}========================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.RED}  🚨 CRITICAL DATA DRIFT ALERT TRIGGERED!                               {Style.RESET}")
    print(f"{Style.BOLD}{Style.RED}========================================================================{Style.RESET}")
    print(f"Batch Status:       {Style.BOLD}{Style.RED}{rep_drifted.overall_status}{Style.RESET}")
    print(f"Breached Features:  {Style.BOLD}{Style.RED}{rep_drifted.drifted_features_count}{Style.RESET} out of {rep_drifted.total_features_monitored} monitored features\n")

    print(f"{Style.BOLD}{'FEATURE':<22} | {'REF MEAN':>9} | {'CURR MEAN':>9} | {'KS STAT':>7} | {'KS P-VAL':>8} | {'PSI':>7} | {'STATUS'}{Style.RESET}")
    print(f"{Style.DIM}{'-' * 84}{Style.RESET}")

    for feat_name, res in rep_drifted.feature_results.items():
        if res.status in ["ALERT", "WARNING"]:
            stat_color = Style.RED if res.status == "ALERT" else Style.YELLOW
            print(
                f"{res.feature_name:<22} | {res.reference_mean:>9.2f} | {res.current_mean:>9.2f} | "
                f"{res.ks_statistic:>7.3f} | {res.ks_p_value:>8.4f} | {res.psi:>7.3f} | {stat_color}{Style.BOLD}{res.status}{Style.RESET}"
            )

    print(f"{Style.DIM}{'-' * 84}{Style.RESET}\n")
    print(f"{Style.BOLD}{Style.YELLOW}[AUTOMATED SYSTEM ACTION RECOMMENDED]{Style.RESET}")
    print(f"  • Trigger model retraining (Module 4) using newly collected production window.")
    print(f"  • Flag incoming high-drift transactions for heightened fraud analyst review (Module 7).")

    print(f"\n{Style.DIM}{'-' * 86}{Style.RESET}")
    print(f"{Style.BOLD}{Style.GREEN}[OK] Module 5 Demo Completed Successfully!{Style.RESET}")
    print(f"\n{Style.DIM}Ready to proceed to Module 6 (Explainability Layer: SHAP Feature Attribution).{Style.RESET}\n")


def run_module_6_demo():
    """Demonstrate Module 6: Explainability Layer with SHAP Feature Attributions."""
    from src.explain.explainer import FraudExplainer
    from src.features.pipeline import FeatureEngineer
    from src.models.train import MODELS_DIR

    print(f"\n{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}  MODULE 6 DEMO: EXPLAINABILITY LAYER (SHAP ATTRIBUTION WATERFALL)       {Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}\n")

    explainer = FraudExplainer(models_dir=MODELS_DIR)
    streamer = TransactionStreamer(DEFAULT_CSV_PATH, delay_seconds=0.0)
    engineer = FeatureEngineer(window_seconds=3600.0)

    print(f"{Style.BOLD}Active Champion Model:{Style.RESET} {explainer.metadata.get('model_type')} ({explainer.metadata.get('version')})")
    print(f"{Style.BOLD}SHAP Explainer Engine:{Style.RESET} TreeExplainer (Exact Tree-Path Marginal Attributions)")
    print(f"{Style.BOLD}Monitored Features:{Style.RESET}   {len(explainer.feature_names)} features\n")

    print(f"{Style.BOLD}--- Step 1: Streaming Transactions to Locate Flagged Fraud Event ---{Style.RESET}")
    flagged_event = None
    flagged_features = None
    legit_event = None
    legit_features = None

    for evt in streamer.stream(max_events=1000):
        enr = engineer.process_transaction(evt, persist=False)
        feat_dict = enr.get_feature_matrix_row()
        
        # Explain in real time
        exp = explainer.explain_transaction(feat_dict, transaction_id=evt.transaction_id)
        if exp.is_fraud_flagged and flagged_event is None:
            flagged_event = evt
            flagged_features = feat_dict
            flagged_explanation = exp

        if not exp.is_fraud_flagged and legit_event is None:
            legit_event = evt
            legit_features = feat_dict

        if flagged_event is not None and legit_event is not None:
            break

    # If dataset has no fraud in first 1000, synthesize a probe
    if flagged_event is None:
        print("Synthesizing anomalous transaction probe for demonstration...")
        sample_fraud_feats = {col: 0.0 for col in explainer.feature_names}
        sample_fraud_feats["amount"] = 920.0
        sample_fraud_feats["spending_deviation"] = 8.5
        sample_fraud_feats["V4"] = 3.6
        sample_fraud_feats["V11"] = 3.1
        sample_fraud_feats["V12"] = -4.2
        sample_fraud_feats["V14"] = -4.8
        flagged_explanation = explainer.explain_transaction(sample_fraud_feats, transaction_id="tx_fraud_probe_01")
    else:
        print(f"Found flagged transaction in stream: {flagged_event.transaction_id}")

    # Step 2: Render local explanation for flagged transaction
    print(f"\n{Style.BOLD}{Style.RED}========================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.RED}  TRANSACTION FLAGGED AS FRAUD: LOCAL SHAP WATERFALL BREAKDOWN          {Style.RESET}")
    print(f"{Style.BOLD}{Style.RED}========================================================================{Style.RESET}")
    print(f"Transaction ID:    {Style.BOLD}{flagged_explanation.transaction_id}{Style.RESET}")
    print(f"Model Risk Score:  {Style.BOLD}{Style.RED}{flagged_explanation.prediction_prob * 100:.2f}% P(Fraud){Style.RESET}")
    print(f"Audit Status:      {Style.BOLD}{Style.RED}SUSPICIOUS / SENT TO REVIEW QUEUE{Style.RESET}\n")

    print(f"{Style.BOLD}{'FEATURE NAME':<22} | {'ACTUAL VAL':>10} | {'SHAP ATTRIBUTION':>17} | {'RISK DIRECTION'}{Style.RESET}")
    print(f"{Style.DIM}{'-' * 78}{Style.RESET}")

    for c in flagged_explanation.top_contributors[:8]:
        if c.shap_value > 0:
            val_col = f"{Style.RED}+{c.shap_value:.4f}{Style.RESET}"
            bar = f"{Style.RED}" + ("#" * max(1, min(18, int(abs(c.shap_value) * 10)))) + f"{Style.RESET}"
            direction = f"{Style.RED}--> +INCREASES FRAUD RISK{Style.RESET}"
        else:
            val_col = f"{Style.GREEN}{c.shap_value:.4f}{Style.RESET}"
            bar = f"{Style.GREEN}" + ("." * max(1, min(18, int(abs(c.shap_value) * 10)))) + f"{Style.RESET}"
            direction = f"{Style.GREEN}<-- -SHIELDS / LEGIT{Style.RESET}"

        print(f"{c.feature_name:<22} | {c.feature_value:>10.2f} | {val_col:>26} {bar:<10} | {direction}")

    print(f"{Style.DIM}{'-' * 78}{Style.RESET}")
    print(f"\n{Style.BOLD}Auditor Natural Language Summary:{Style.RESET}")
    print(f"  {Style.CYAN}{flagged_explanation.summary_reason}{Style.RESET}")

    # Step 3: Contrast with legitimate transaction
    print(f"\n{Style.BOLD}--- Step 3: Contrast with Legitimate Transaction ({legit_event.transaction_id}) ---{Style.RESET}")
    legit_exp = explainer.explain_transaction(legit_features, transaction_id=legit_event.transaction_id)
    print(f"Transaction ID:   {legit_exp.transaction_id} (Amount: ${legit_event.amount:.2f})")
    print(f"Model Risk Score: {Style.BOLD}{Style.GREEN}{legit_exp.prediction_prob * 100:.2f}% P(Fraud){Style.RESET} [PASSED]")
    print(f"Top Attributions:")
    for c in legit_exp.top_contributors[:3]:
        print(f"  • {c.feature_name:<18} (val: {c.feature_value:>6.2f}) -> SHAP: {c.shap_value:+.4f} ({c.impact})")

    print(f"\n{Style.DIM}{'-' * 86}{Style.RESET}")
    print(f"{Style.BOLD}{Style.GREEN}[OK] Module 6 Demo Completed Successfully!{Style.RESET}")
    print(f"\n{Style.DIM}Ready to proceed to Module 7 (Case Management Dashboard: Streamlit Reviewer Portal).{Style.RESET}\n")


def run_module_7_demo():
    """Demonstrate Module 7: Case Management & Reviewer Decision Storage."""
    from db.storage import FeatureStoreDB, DEFAULT_DB_PATH
    from src.explain.explainer import FraudExplainer
    from src.features.pipeline import FeatureEngineer

    print(f"\n{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}  MODULE 7 DEMO: CASE MANAGEMENT & REVIEWER TRIAGE DASHBOARD            {Style.RESET}")
    print(f"{Style.BOLD}{Style.CYAN}========================================================================{Style.RESET}\n")

    db = FeatureStoreDB(db_path=DEFAULT_DB_PATH)
    explainer = FraudExplainer(models_dir=MODELS_DIR)
    streamer = TransactionStreamer(DEFAULT_CSV_PATH, delay_seconds=0.0)
    engineer = FeatureEngineer(window_seconds=3600.0)

    # Step 1: Stream transactions and populate triage queue with flagged cases
    print(f"{Style.BOLD}--- Step 1: Streaming Transactions to Populate Triage Queue ---{Style.RESET}")
    flagged_count = 0

    for evt in streamer.stream(max_events=120):
        enr = engineer.process_transaction(evt, persist=True)
        feat_dict = enr.get_feature_matrix_row()
        exp = explainer.explain_transaction(feat_dict, transaction_id=evt.transaction_id)

        if exp.is_fraud_flagged:
            db.save_flagged_case({
                "transaction_id": evt.transaction_id,
                "user_id": evt.user_id,
                "timestamp": evt.timestamp,
                "amount": evt.amount,
                "risk_score": exp.prediction_prob,
                "status": "PENDING",
                "shap_summary": exp.summary_reason,
                "shap_features": exp.to_dict()["top_contributors"],
            })
            flagged_count += 1
            if flagged_count >= 2:
                break

    # If stream didn't hit 2 natural frauds, add realistic probe cases
    if flagged_count < 2:
        for idx in range(flagged_count + 1, 3):
            tx_id = f"tx_case_demo_{idx:03d}"
            probe_feats = {col: 0.0 for col in explainer.feature_names}
            probe_feats["amount"] = 890.0 * idx
            probe_feats["spending_deviation"] = 7.5 * idx
            probe_feats["V4"] = 3.8
            probe_feats["V12"] = -4.1
            exp = explainer.explain_transaction(probe_feats, transaction_id=tx_id)
            db.save_flagged_case({
                "transaction_id": tx_id,
                "user_id": f"usr_{idx * 111:04d}",
                "timestamp": f"2026-09-06 01:{30 + idx * 5:02d}:00",
                "amount": probe_feats["amount"],
                "risk_score": exp.prediction_prob,
                "status": "PENDING",
                "shap_summary": exp.summary_reason,
                "shap_features": exp.to_dict()["top_contributors"],
            })
            flagged_count += 1

    stats_initial = db.get_reviewer_stats()
    print(f"Queue Status: {stats_initial['total_flagged']} Total Flagged | {stats_initial['pending']} Pending Review\n")

    # Step 2: Simulate reviewer triage on Case 1
    pending_cases = db.get_flagged_cases(status_filter="PENDING")
    case_1 = pending_cases[0]
    print(f"{Style.BOLD}--- Step 2: Reviewer Inspecting Case 1 ({case_1['transaction_id']}) ---{Style.RESET}")
    print(f"  • User ID:         {case_1['user_id']}")
    print(f"  • Amount:          ${case_1['amount']:.2f}")
    print(f"  • Model Risk:      {Style.RED}{case_1['risk_score'] * 100:.2f}% P(Fraud){Style.RESET}")
    print(f"  • AI Explanation:  {Style.CYAN}{case_1['shap_summary']}{Style.RESET}")

    print(f"\n  [ACTION: Reviewer Clicks: {Style.RED}{Style.BOLD}'CONFIRMED FRAUD'{Style.RESET}]")
    db.record_reviewer_decision(
        transaction_id=case_1["transaction_id"],
        decision="CONFIRMED_FRAUD",
        notes="Cardholder confirmed unauthorized remote terminal charge."
    )
    print(f"  {Style.GREEN}✔ Decision committed to SQLite! Status updated to CONFIRMED_FRAUD.{Style.RESET}\n")

    # Step 3: Simulate reviewer triage on Case 2
    if len(pending_cases) > 1:
        case_2 = pending_cases[1]
        print(f"{Style.BOLD}--- Step 3: Reviewer Inspecting Case 2 ({case_2['transaction_id']}) ---{Style.RESET}")
        print(f"  • User ID:         {case_2['user_id']}")
        print(f"  • Amount:          ${case_2['amount']:.2f}")
        print(f"  • Model Risk:      {Style.YELLOW}{case_2['risk_score'] * 100:.2f}% P(Fraud){Style.RESET}")
        print(f"\n  [ACTION: Reviewer Clicks: {Style.GREEN}{Style.BOLD}'FALSE ALARM'{Style.RESET}]")
        db.record_reviewer_decision(
            transaction_id=case_2["transaction_id"],
            decision="FALSE_ALARM",
            notes="Legitimate high-value purchase verified via phone call."
        )
        print(f"  {Style.GREEN}✔ Decision committed to SQLite! Status updated to FALSE_ALARM.{Style.RESET}\n")

    # Step 4: Verify decisions in SQLite audit trail
    print(f"{Style.BOLD}--- Step 4: Verifying Reviewer Decision Audit Trail in SQLite ---{Style.RESET}")
    history = db.get_decision_history(limit=5)
    print(f"{Style.BOLD}{'DECISION ID':<12} | {'TX ID':<16} | {'DECISION':<16} | {'REVIEWER NOTES'}{Style.RESET}")
    print(f"{Style.DIM}{'-' * 84}{Style.RESET}")
    for h in history[:3]:
        dec_color = Style.RED if h['decision'] == "CONFIRMED_FRAUD" else Style.GREEN
        print(f"{h['decision_id']:<12} | {h['transaction_id']:<16} | {dec_color}{h['decision']:<16}{Style.RESET} | {h['notes']}")

    print(f"{Style.DIM}{'-' * 84}{Style.RESET}\n")
    final_stats = db.get_reviewer_stats()
    print(f"{Style.BOLD}Updated Case Summary:{Style.RESET}")
    print(f"  • Pending Review:   {final_stats['pending']}")
    print(f"  • Confirmed Fraud:  {Style.RED}{final_stats['confirmed_fraud']}{Style.RESET}")
    print(f"  • False Alarms:     {Style.GREEN}{final_stats['false_alarms']}{Style.RESET}")

    print(f"\n{Style.BOLD}--- Interactive Streamlit Web Dashboard ---{Style.RESET}")
    print(f"Launch the interactive dashboard anytime using:")
    print(f"  {Style.BOLD}{Style.CYAN}.\\.venv\\Scripts\\streamlit run dashboard/app.py{Style.RESET}")

    print(f"\n{Style.DIM}{'-' * 86}{Style.RESET}")
    print(f"{Style.BOLD}{Style.GREEN}[OK] Module 7 Demo Completed Successfully!{Style.RESET}")
    print(f"\n{Style.DIM}Ready to proceed to Module 8 (Feedback Loop & Closed-Loop End-to-End Integration).{Style.RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Fraud Detection System Demo Runner")
    parser.add_argument("--module", type=int, default=1, choices=range(1, 9), help="Module number to run (1-8)")
    parser.add_argument("--limit", type=int, default=20, help="Max transactions to process in demo")
    parser.add_argument("--delay", type=float, default=0.06, help="Simulated streaming delay in seconds")
    parser.add_argument("--mode", type=str, default="generator", choices=["generator", "queue"], help="Streamer engine mode")
    args = parser.parse_args()

    if args.module == 1:
        run_module_1_demo(max_events=args.limit, delay=args.delay, mode=args.mode)
    elif args.module == 2:
        run_module_2_demo(tx_limit=args.limit if args.limit <= 5 else 3)
    elif args.module == 3:
        run_module_3_demo(max_rows=5000)
    elif args.module == 4:
        run_module_4_demo()
    elif args.module == 5:
        run_module_5_demo()
    elif args.module == 6:
        run_module_6_demo()
    elif args.module == 7:
        run_module_7_demo()
    else:
        print(f"{Style.YELLOW}Module {args.module} is scheduled for subsequent implementation.{Style.RESET}")
        print("We are strictly developing incrementally.")


if __name__ == "__main__":
    main()






