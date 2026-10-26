"""Interactive Streamlit Replay Dashboard for Sentinel Anomaly Detection.

Features:
- Stream series replay with live timeline visualization
- Ground truth anomaly windows vs predicted alerts overlay
- Anomaly score and dynamic threshold tracking
- Detailed evidence inspector card for each detected event
- Model benchmark comparison scorecard (F1, Precision, Recall, Latency, Throughput)
"""
from __future__ import annotations

import json
import os
import time
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sentinel.config import AdaptationConfig, DetectorConfig, SentinelConfig
from sentinel.data.loader import NABDataLoader
from sentinel.evaluation.event_f1 import evaluate_event_f1
from sentinel.explanations.explainer import generate_explanation
from sentinel.pipeline import SentinelPipeline

st.set_page_config(page_title="Sentinel - Adaptive Anomaly Detection", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 10px;
    }
    .alert-card {
        background-color: #2b1f24;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #f44336;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Sentinel: Adaptive Streaming Time-Series Anomaly Detector")
st.caption("Utkrishta 2026 AI/ML Hackathon — Challenge 2 (Adaptive Time-Series Anomaly Detection)")


@st.cache_resource
def get_loader():
    return NABDataLoader()


loader = get_loader()
series_list = loader.list_series()
gt_windows = loader.get_ground_truth_windows_by_series_id()

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
detector_choice = st.sidebar.selectbox(
    "Active Detector",
    ["Hybrid Ensemble (TCN-AE + MAD + EWMA)", "Causal TCN-AE (GPU)", "Rolling Median + MAD", "Streaming EWMA", "Isolation Forest"],
)

det_type_map = {
    "Hybrid Ensemble (TCN-AE + MAD + EWMA)": "ensemble",
    "Causal TCN-AE (GPU)": "autoencoder",
    "Rolling Median + MAD": "rolling_mad",
    "Streaming EWMA": "ewma",
    "Isolation Forest": "iforest",
}

threshold_mode = st.sidebar.selectbox("Thresholding Logic", ["likelihood", "robust_z"])
enter_th = st.sidebar.slider("Enter Threshold", 1.5, 12.0, 6.5, 0.1)
exit_th = st.sidebar.slider("Exit Threshold (Hysteresis)", 1.0, 6.0, 2.5, 0.1)
drift_guard = st.sidebar.checkbox("Enable ADWIN Drift Guard", value=True)

# Series selection
series_options = {f"{s.series_id}: [{s.category}] {s.series_name}": s.series_id for s in series_list}
selected_label = st.sidebar.selectbox("Benchmark Series", list(series_options.keys()), index=11)
selected_sid = series_options[selected_label]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Stream Replay & Inspection", "📈 Benchmark Scorecard", "🔬 System Architecture & Explainability"])

with tab1:
    col_ctrl1, col_ctrl2 = st.columns([1, 4])
    with col_ctrl1:
        run_btn = st.button("▶️ Execute Stream Analysis", type="primary")

    if run_btn:
        with st.spinner("Streaming observations through Sentinel pipeline..."):
            det_cfg = DetectorConfig(
                detector_type=det_type_map[detector_choice],
                ae_device="cuda",
                scale_floor=0.05,
            )
            adapt_cfg = AdaptationConfig(
                enter_threshold=enter_th,
                exit_threshold=exit_th,
                threshold_mode=threshold_mode,
                drift_enabled=drift_guard,
                min_event_length=3,
                cooldown_steps=10,
            )
            pipeline = SentinelPipeline(SentinelConfig(detector=det_cfg, adaptation=adapt_cfg))

            stream = loader.stream_series(selected_sid)
            records, stats = pipeline.process_stream(stream)

            # Compute metrics for this series
            sub_gt = {selected_sid: gt_windows.get(selected_sid, [])}
            eval_metrics = evaluate_event_f1(records, sub_gt)

            # Metrics bar
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Event-Based F1", f"{eval_metrics.event_f1:.3f}")
            m2.metric("Precision / Recall", f"{eval_metrics.precision:.2f} / {eval_metrics.recall:.2f}")
            m3.metric("True Positives / Events", f"{eval_metrics.true_positives} / {eval_metrics.total_true_events}")
            m4.metric("Throughput", f"{stats['throughput_points_per_sec']:.0f} pts/sec")
            m5.metric("Latency", f"{stats['latency_ms_per_point']:.2f} ms/pt")

            # Prepare plot dataframe
            plot_rows = []
            for r in records:
                ev = r["evidence"]
                plot_rows.append({
                    "timestamp": pd.to_datetime(r["timestamp"]),
                    "observed": ev["observed"],
                    "baseline": ev["baseline"],
                    "score": r["anomaly_score"],
                    "threshold": r["threshold"],
                    "label": r["label"],
                    "event_id": r["event_id"],
                })
            df_plot = pd.DataFrame(plot_rows)

            # Time series chart
            st.subheader("Time-Series & Detection Stream")
            base_chart = alt.Chart(df_plot).encode(x="timestamp:T")

            line_obs = base_chart.mark_line(color="#2196F3", opacity=0.8).encode(
                y=alt.Y("observed:Q", title="Value / Baseline"),
                tooltip=["timestamp:T", "observed:Q", "baseline:Q", "score:Q", "label:N"]
            )
            line_base = base_chart.mark_line(color="#4CAF50", strokeDash=[4, 4], opacity=0.6).encode(
                y="baseline:Q"
            )
            points_anomaly = base_chart.transform_filter(
                alt.datum.label == "ANOMALOUS"
            ).mark_circle(size=60, color="#FF5722").encode(
                y="observed:Q",
                tooltip=["timestamp:T", "observed:Q", "score:Q", "event_id:N"]
            )

            chart_ts = alt.layer(line_obs, line_base, points_anomaly).properties(height=320)
            st.altair_chart(chart_ts, use_container_width=True)

            # Score & Threshold chart
            st.subheader("Dynamic Anomaly Score & Hysteresis Threshold")
            line_score = base_chart.mark_line(color="#E91E63").encode(
                y=alt.Y("score:Q", title="Anomaly Score"),
                tooltip=["timestamp:T", "score:Q", "threshold:Q"]
            )
            line_thresh = base_chart.mark_line(color="#FFC107", strokeDash=[3, 3]).encode(
                y="threshold:Q"
            )
            chart_score = alt.layer(line_score, line_thresh).properties(height=180)
            st.altair_chart(chart_score, use_container_width=True)

            # Evidence cards for detected events
            st.subheader("🔍 Interpretable Anomaly Evidence Trail")
            anom_records = [r for r in records if r["label"] == "ANOMALOUS"]
            if anom_records:
                unique_events = {}
                for r in anom_records:
                    eid = r.get("event_id")
                    if eid and eid not in unique_events:
                        unique_events[eid] = r

                for eid, r in list(unique_events.items())[:5]:
                    expl = generate_explanation(r)
                    with st.expander(f"⚠️ Event [{eid}] at {r['timestamp']} - Score: {r['anomaly_score']:.2f}"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Observed", f"{expl['observed_value']:.2f}")
                        c2.metric("Expected Baseline", f"{expl['expected_baseline']:.2f}")
                        c3.metric("Residual Delta", f"{expl['residual']:+.2f} ({expl['percent_deviation']:+.1f}%)")
                        st.info(f"**Explanation:** {expl['narrative']}")
                        st.write(f"**Top Contributing Detectors:** {', '.join(expl['top_contributors'])}")
                        st.write(f"**Trigger Reason Codes:** `{', '.join(expl['reason_codes'])}`")
            else:
                st.success("No anomalies detected in this stream segment.")

with tab2:
    st.subheader("🏆 Model Benchmark Scorecard (Utkrishta 2026 Evaluation Protocol)")
    if os.path.exists("outputs/experiment_log.csv"):
        exp_df = pd.read_csv("outputs/experiment_log.csv")
        st.dataframe(exp_df, use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.bar_chart(exp_df.set_index("model_name")["event_f1"])
        with col_b2:
            st.bar_chart(exp_df.set_index("model_name")["throughput_pts_sec"])
    else:
        st.info("Run `python scripts/run_experiments.py` to generate multi-model benchmark results.")

with tab3:
    st.subheader("Architecture & Design Guarantees")
    st.markdown("""
    - **Strict Causality**: Window features and statistics rely exclusively on past observations $t-1, \dots, t-W$. No future data leakage.
    - **Guarded Adaptation**: Contaminated baseline prevention: anomalous observations are rejected from updating baseline buffers unless regime shift threshold is exceeded.
    - **ADWIN Drift Engine**: Automatically cuts older reference samples when distribution changes without arbitrary time window guessing.
    - **Hysteresis Thresholding**: Two-tier thresholding prevents alert flickering. An event enters at $\tau_{enter}$ and only exits below $\tau_{exit}$.
    - **RTX 5050 Native Acceleration**: Deep Causal TCN-Autoencoder executes on Blackwell CUDA tensors with sub-millisecond inference latency.
    """)
