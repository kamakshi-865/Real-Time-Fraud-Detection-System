"""
Module 7: Interactive Case Management Dashboard.
Streamlit application for financial fraud analysts to inspect flagged transactions,
visualize local SHAP feature attributions, and record triage decisions into SQLite.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import streamlit as st

# Setup project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.storage import FeatureStoreDB, DEFAULT_DB_PATH
from src.explain.explainer import FraudExplainer
from src.ingestion.streamer import TransactionStreamer
from src.features.pipeline import FeatureEngineer
from data.setup_data import DEFAULT_CSV_PATH
from src.models.retrain import ModelRegistry, MODELS_DIR

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Fraud Sentinel | Case Management Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern financial portal aesthetic
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-fraud {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-pending {
        background-color: #f59e0b;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-legit {
        background-color: #10b981;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Load DB manager, explainer, and model registry."""
    db = FeatureStoreDB(db_path=DEFAULT_DB_PATH)
    explainer = FraudExplainer(models_dir=MODELS_DIR)
    registry = ModelRegistry(models_dir=MODELS_DIR)
    return db, explainer, registry


db, explainer, registry = get_services()


# --- Sidebar ---
st.sidebar.markdown("### 🛡️ Sentinel System Monitor")
active_ver = registry._load_registry().get("active_version", "v1")
st.sidebar.info(f"**Active Model:** XGBoost ({active_ver})\n\n**Feature Store:** SQLite ({DEFAULT_DB_PATH.name})")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Simulation Controls")

if st.sidebar.button("⚡ Ingest & Score Stream (30 Txs)", use_container_width=True):
    with st.spinner("Streaming & scoring real-time transactions..."):
        streamer = TransactionStreamer(DEFAULT_CSV_PATH, delay_seconds=0.0)
        engineer = FeatureEngineer(window_seconds=3600.0)
        new_flagged = 0

        # Stream transactions from offset
        for evt in streamer.stream(max_events=30, start_row=50):
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
                    "shap_features": [asdict_c for asdict_c in exp.to_dict()["top_contributors"]],
                })
                new_flagged += 1

        # If none flagged naturally, generate a suspicious probe for reviewer testing
        if new_flagged == 0:
            probe_id = f"tx_probe_{len(db.get_flagged_cases()) + 1:03d}"
            probe_feats = {col: 0.0 for col in explainer.feature_names}
            probe_feats["amount"] = 875.50
            probe_feats["spending_deviation"] = 9.2
            probe_feats["V4"] = 4.1
            probe_feats["V12"] = -3.9
            exp = explainer.explain_transaction(probe_feats, transaction_id=probe_id)
            db.save_flagged_case({
                "transaction_id": probe_id,
                "user_id": "usr_0421",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "amount": 875.50,
                "risk_score": exp.prediction_prob,
                "status": "PENDING",
                "shap_summary": exp.summary_reason,
                "shap_features": exp.to_dict()["top_contributors"],
            })
            new_flagged = 1

    st.sidebar.success(f"Stream processed! Flagged {new_flagged} suspicious cases.")
    st.rerun()

status_filter = st.sidebar.selectbox(
    "Filter Queue by Status:",
    ["PENDING", "ALL", "CONFIRMED_FRAUD", "FALSE_ALARM"],
    index=0
)

# --- Header & Metrics ---
st.markdown('<div class="main-header">Fraud Case Management & Reviewer Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Human-in-the-Loop Triage, SHAP Attribution Analysis, and Feedback Recording</div>', unsafe_allow_html=True)

stats = db.get_reviewer_stats()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f8fafc;">{stats["total_flagged"]}</div><div class="metric-label">Total Flagged</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#f59e0b;">{stats["pending"]}</div><div class="metric-label">Pending Review</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444;">{stats["confirmed_fraud"]}</div><div class="metric-label">Confirmed Fraud</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#10b981;">{stats["false_alarms"]}</div><div class="metric-label">False Alarms</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Fetch cases
cases = db.get_flagged_cases(status_filter=status_filter)

if not cases:
    st.info("No cases matching the selected filter. Click **'Ingest & Score Stream'** in the sidebar to simulate live transactions!")
else:
    # Split layout: Queue table on left, Case detail & SHAP explanation on right
    left_col, right_col = st.columns([1, 1.4])

    with left_col:
        st.markdown("### 📋 Triage Queue")
        
        # Build display table
        display_data = []
        for c in cases:
            display_data.append({
                "Tx ID": c["transaction_id"],
                "User": c["user_id"],
                "Amount": f"${c['amount']:.2f}",
                "Risk": f"{c['risk_score']*100:.1f}%",
                "Status": c["status"],
            })
        df_queue = pd.DataFrame(display_data)
        st.dataframe(df_queue, use_container_width=True, hide_index=True)

        case_options = [c["transaction_id"] for c in cases]
        selected_tx = st.selectbox("Select Transaction to Inspect:", case_options, index=0)

    # Find chosen case
    selected_case = next(c for c in cases if c["transaction_id"] == selected_tx)

    with right_col:
        st.markdown(f"### 🔍 Case Detail: `{selected_case['transaction_id']}`")
        
        # Risk header badge
        risk_pct = selected_case["risk_score"] * 100
        badge_style = "badge-fraud" if risk_pct > 80 else ("badge-pending" if risk_pct > 50 else "badge-legit")
        st.markdown(
            f"**User ID:** `{selected_case['user_id']}` &nbsp;|&nbsp; "
            f"**Amount:** `${selected_case['amount']:.2f}` &nbsp;|&nbsp; "
            f"**Timestamp:** `{selected_case['timestamp']}` &nbsp;|&nbsp; "
            f"**Status:** <span class='{badge_style}'>{selected_case['status']}</span>",
            unsafe_allow_html=True
        )

        st.progress(min(1.0, selected_case["risk_score"]))
        st.caption(f"Predicted Fraud Probability: **{risk_pct:.2f}%**")

        # SHAP Waterfall / Attribution Breakdown
        st.markdown("#### 📊 SHAP Feature Attribution Breakdown")
        shap_items = selected_case.get("shap_features", [])

        if shap_items:
            # Prepare top 8 features for bar chart
            top_shap = shap_items[:8]
            chart_df = pd.DataFrame({
                "Feature": [item["feature_name"] for item in top_shap],
                "SHAP Value": [item["shap_value"] for item in top_shap],
                "Actual Value": [item["feature_value"] for item in top_shap],
                "Direction": ["Increases Fraud Risk" if item["shap_value"] > 0 else "Shields / Legit" for item in top_shap]
            })

            # Horizontal bar visualization
            import altair as alt
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("SHAP Value:Q", title="SHAP Value (Impact on Model Log-Odds)"),
                y=alt.Y("Feature:N", sort="-x", title="Feature"),
                color=alt.Color("Direction:N", scale=alt.Scale(
                    domain=["Increases Fraud Risk", "Shields / Legit"],
                    range=["#ef4444", "#10b981"]
                )),
                tooltip=["Feature", "SHAP Value", "Actual Value", "Direction"]
            ).properties(height=280)

            st.altair_chart(chart, use_container_width=True)

        # Natural language summary
        if selected_case.get("shap_summary"):
            st.info(f"💡 **AI Auditor Explanation:** {selected_case['shap_summary']}")

        # --- Reviewer Action Buttons ---
        st.markdown("#### ✍️ Reviewer Triage Action")
        notes_input = st.text_input(
            "Reviewer Notes / Evidence:",
            value=selected_case.get("reviewer_notes") or "",
            placeholder="e.g., Confirmed with cardholder via SMS; unauthorized charge."
        )

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            if st.button("🚨 Confirmed Fraud", type="primary", use_container_width=True):
                db.record_reviewer_decision(
                    transaction_id=selected_case["transaction_id"],
                    decision="CONFIRMED_FRAUD",
                    notes=notes_input or "Marked as confirmed fraud by reviewer."
                )
                st.success(f"Case `{selected_case['transaction_id']}` saved as CONFIRMED FRAUD in SQLite!")
                st.rerun()

        with btn_col2:
            if st.button("🛡️ False Alarm", type="secondary", use_container_width=True):
                db.record_reviewer_decision(
                    transaction_id=selected_case["transaction_id"],
                    decision="FALSE_ALARM",
                    notes=notes_input or "Verified legitimate activity by cardholder."
                )
                st.info(f"Case `{selected_case['transaction_id']}` saved as FALSE ALARM in SQLite!")
                st.rerun()

# --- Chronological Reviewer Audit Trail ---
st.markdown("---")
st.markdown("### 📜 Reviewer Decision Audit Trail (SQLite Persistent Log)")
audit_records = db.get_decision_history(limit=10)

if audit_records:
    df_audit = pd.DataFrame(audit_records)
    st.dataframe(df_audit[["decision_id", "transaction_id", "user_id", "amount", "risk_score", "decision", "notes", "reviewed_at"]], use_container_width=True, hide_index=True)
else:
    st.caption("No decisions logged yet. Use the review buttons above to triage cases.")
