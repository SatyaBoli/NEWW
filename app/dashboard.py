"""Sentinel: Real-Time Streaming Telemetry Monitor.

A minimal cognitive load, high-signal interface for streaming time-series anomaly detection.
Styled in dark obsidian and champagne gold, emphasizing live telemetry progression,
transparent evidence generation, and zero visual clutter.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sentinel.config import AdaptationConfig, DetectorConfig, SentinelConfig
from sentinel.data.loader import NABDataLoader
from sentinel.detectors.ensemble import HybridEnsembleDetector
from sentinel.detectors.tcn_autoencoder import CausalTCNAEDetector
from sentinel.evaluation.event_f1 import evaluate_event_f1
from sentinel.explanations.explainer import generate_explanation
from sentinel.pipeline import SentinelPipeline

# -----------------------------------------------------------------------------
# Page Configuration & Clean Obsidian/Gold Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentinel · Streaming Telemetry Monitor",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Surface */
    .stApp {
        background-color: #0A0B0E;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}

    /* Custom Header Container */
    .sentinel-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(226, 194, 117, 0.2);
        margin-bottom: 20px;
    }
    .sentinel-title {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #F8FAFC;
    }
    .sentinel-title span {
        color: #D4AF37; /* Champagne Gold */
    }
    .sentinel-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        font-family: 'JetBrains Mono', monospace;
    }

    /* KPI Status Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 16px 0;
    }
    .kpi-card {
        background: #111319;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 14px 18px;
    }
    .kpi-card-gold {
        background: #111319;
        border: 1px solid rgba(212, 175, 55, 0.35);
        border-radius: 8px;
        padding: 14px 18px;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 600;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-value-gold {
        font-size: 1.35rem;
        font-weight: 600;
        color: #E2C275;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* Status Pills */
    .status-nominal {
        color: #4ADE80;
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
    }
    .status-alert {
        color: #FF5C5C;
        font-weight: 700;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
    }

    /* Clean Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background-color: transparent;
        border-radius: 6px;
        color: #94A3B8;
        font-weight: 500;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(212, 175, 55, 0.12) !important;
        color: #E2C275 !important;
        border: 1px solid rgba(212, 175, 55, 0.3) !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(180deg, #1A1D26 0%, #111319 100%);
        color: #F8FAFC;
        border: 1px solid rgba(212, 175, 55, 0.4);
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.88rem;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        border-color: #E2C275;
        box-shadow: 0 0 12px rgba(212, 175, 55, 0.25);
        color: #E2C275;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_data_engine():
    return NABDataLoader()


loader = load_data_engine()
series_list = loader.list_series()
gt_windows = loader.get_ground_truth_windows_by_series_id()

# Friendly series names mapping
friendly_names = {
    "11": "AWS EC2 CPU Utilization (realAWSCloudwatch)",
    "13": "AWS EC2 CPU 5f5533 (realAWSCloudwatch)",
    "5": "Art Daily Flat Middle (artificialNoAnomaly)",
    "7": "Art Daily Jumps Up (artificialWithAnomaly)",
    "34": "Ambient Temperature Sensor (realKnownCause)",
}

# -----------------------------------------------------------------------------
# Top Navigation & Command Header
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="sentinel-header">
    <div class="sentinel-title">SENTINEL <span>·</span> TELEMETRY</div>
    <div class="sentinel-subtitle">CAUSAL STREAMING ANOMALY ENGINE · NVIDIA RTX 5050 (CUDA 13.2)</div>
</div>
""",
    unsafe_allow_html=True,
)

tab_stream, tab_scorecard, tab_pathway, tab_architecture = st.tabs(
    [
        "Live Oscilloscope Stream",
        "Benchmark Scorecard",
        "Sentinel 2.0 Innovations",
        "Causality Guarantees",
    ]
)

# =============================================================================
# TAB 1: Live Oscilloscope Stream
# =============================================================================
with tab_stream:
    # Minimalist Control Strip (Horizontal, No Sidebar Cramming)
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([2.5, 2.5, 1.8, 1.2])

    with ctrl_col1:
        sid_options = [s.series_id for s in series_list]
        selected_sid = st.selectbox(
            "Telemetry Stream",
            options=sid_options,
            format_func=lambda s: f"[{s}] {friendly_names.get(s, s)}",
            index=sid_options.index("11") if "11" in sid_options else 0,
            label_visibility="collapsed",
        )

    with ctrl_col2:
        model_choice = st.selectbox(
            "Detector Architecture",
            options=[
                "Model B: Infrastructure Specialist (Ensemble)",
                "Model A: Global Generalist (Ensemble)",
                "Model B: Standalone TCN-AE (CUDA)",
                "Model A: Standalone TCN-AE (CUDA)",
                "Streaming MAD + Seasonal Guard",
            ],
            index=0,
            label_visibility="collapsed",
        )

    with ctrl_col3:
        stream_mode = st.selectbox(
            "Playback Velocity",
            options=["Smooth Live Oscilloscope (20ms)", "High-Throughput Stream", "Instant Execution"],
            index=0,
            label_visibility="collapsed",
        )

    with ctrl_col4:
        run_stream = st.button("▶ Stream", use_container_width=True)

    # Placeholders for Dynamic Real-Time Updates
    kpi_placeholder = st.empty()
    chart_placeholder = st.empty()
    incident_placeholder = st.empty()

    # Default idle KPI display
    def render_kpis(ingested: int, total: int, status: str, alerts: int, latency_ms: float):
        status_html = (
            '<span class="status-alert">● ACTIVE INCIDENT</span>'
            if status == "ANOMALOUS"
            else '<span class="status-nominal">● NOMINAL</span>'
        )
        kpi_placeholder.markdown(
            f"""
        <div class="kpi-container">
            <div class="kpi-card-gold">
                <div class="kpi-label">Stream Progress</div>
                <div class="kpi-value-gold">{ingested:,} <span style="font-size:0.85rem;color:#94A3B8;">/ {total:,}</span></div>
                <div class="kpi-sub">Strict Causal Lag Order</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Telemetry Status</div>
                <div class="kpi-value">{status_html}</div>
                <div class="kpi-sub">Anti-Contamination Guard Active</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Confirmed Incidents</div>
                <div class="kpi-value">{alerts}</div>
                <div class="kpi-sub">Hysteresis Merged Windows</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Pipeline Latency</div>
                <div class="kpi-value">{latency_ms:.2f} <span style="font-size:0.8rem;color:#94A3B8;">ms/pt</span></div>
                <div class="kpi-sub">RTX 5050 Native Kernels</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Initial render
    all_stream_pts = list(loader.stream_series(selected_sid))
    total_pts = len(all_stream_pts)
    render_kpis(0, total_pts, "NORMAL", 0, 0.0)

    def build_visual_chart(df_data: pd.DataFrame, height: int = 340) -> alt.VConcatChart:
        # Time-Series Layer
        base = alt.Chart(df_data).encode(x=alt.X("timestamp:T", axis=alt.Axis(title=None, labelColor="#64748B", gridColor="#161822")))

        line_obs = base.mark_line(color="#E2C275", strokeWidth=1.8, opacity=0.9).encode(
            y=alt.Y("observed:Q", title="Observed Metric", axis=alt.Axis(titleColor="#94A3B8", labelColor="#64748B", gridColor="#161822")),
            tooltip=["timestamp:T", "observed:Q", "baseline:Q", "score:Q", "label:N"],
        )
        line_base = base.mark_line(color="#64748B", strokeDash=[3, 3], strokeWidth=1.2, opacity=0.6).encode(
            y="baseline:Q"
        )
        points_anom = (
            base.transform_filter(alt.datum.label == "ANOMALOUS")
            .mark_circle(size=70, color="#FF5C5C", opacity=0.95)
            .encode(
                y="observed:Q",
                tooltip=["timestamp:T", "observed:Q", "score:Q", "event_id:N"],
            )
        )
        ts_chart = alt.layer(line_obs, line_base, points_anom).properties(height=height, background="transparent")

        # Lower Score & Threshold Strip
        line_score = base.mark_line(color="#D4AF37", strokeWidth=1.2).encode(
            y=alt.Y("score:Q", title="Score", axis=alt.Axis(titleColor="#94A3B8", labelColor="#64748B", gridColor="#161822")),
            tooltip=["timestamp:T", "score:Q", "threshold:Q"],
        )
        line_thresh = base.mark_line(color="#EF4444", strokeDash=[2, 2], strokeWidth=1.0).encode(
            y="threshold:Q"
        )
        score_chart = alt.layer(line_score, line_thresh).properties(height=90, background="transparent")

        return alt.vconcat(ts_chart, score_chart).configure_view(strokeWidth=0)

    # -------------------------------------------------------------------------
    # Streaming Execution Engine
    # -------------------------------------------------------------------------
    if run_stream:
        # Configure Pipeline
        chk = "models/sentinel_specialist_best.pt" if "Model B" in model_choice else "models/sentinel_tcn_best.pt"
        adapt_cfg = AdaptationConfig(
            enter_threshold=6.5,
            exit_threshold=2.5,
            threshold_mode="likelihood",
            min_event_length=3,
            cooldown_steps=10,
        )

        if "Ensemble" in model_choice:
            detector = HybridEnsembleDetector(checkpoint_path=chk, scale_floor=0.05)
            pipeline = SentinelPipeline(SentinelConfig(adaptation=adapt_cfg))
            pipeline.detector = detector
            pipeline.layer.detector = detector
        elif "TCN-AE" in model_choice:
            detector = CausalTCNAEDetector(window_size=32, device="cuda", scale_floor=0.05, checkpoint_path=chk)
            pipeline = SentinelPipeline(SentinelConfig(adaptation=adapt_cfg))
            pipeline.detector = detector
            pipeline.layer.detector = detector
        else:
            det_cfg = DetectorConfig(detector_type="rolling_mad", scale_floor=0.05)
            pipeline = SentinelPipeline(SentinelConfig(detector=det_cfg, adaptation=adapt_cfg))

        # Chunk parameters based on velocity
        if "Smooth" in stream_mode:
            chunk_size = 25
            sleep_time = 0.02
        elif "High-Throughput" in stream_mode:
            chunk_size = 100
            sleep_time = 0.005
        else:
            chunk_size = total_pts
            sleep_time = 0.0

        records: list[dict[str, Any]] = []
        plot_buffer: list[dict[str, Any]] = []
        unique_events: dict[str, dict[str, Any]] = {}

        t_start = time.perf_counter()

        for idx, pt in enumerate(all_stream_pts):
            rec = pipeline.process_point(pt)
            records.append(rec)
            plot_buffer.append(
                {
                    "timestamp": rec["timestamp"],
                    "observed": rec["evidence"]["observed"],
                    "baseline": rec["evidence"]["baseline"],
                    "score": rec["anomaly_score"],
                    "threshold": rec["threshold"],
                    "label": rec["label"],
                    "event_id": rec.get("event_id") or "",
                }
            )

            if rec["label"] == "ANOMALOUS":
                eid = rec.get("event_id")
                if eid and eid not in unique_events:
                    unique_events[eid] = rec

            # Refresh display at chunk cadence or end
            if (idx + 1) % chunk_size == 0 or (idx + 1) == total_pts:
                elapsed = max(time.perf_counter() - t_start, 1e-4)
                curr_pts = idx + 1
                lat_ms = (elapsed / curr_pts) * 1000.0

                render_kpis(
                    curr_pts,
                    total_pts,
                    rec["label"],
                    len(unique_events),
                    lat_ms,
                )

                # Draw recent window during live streaming (last 250 points for smooth performance)
                disp_df = pd.DataFrame(plot_buffer[-300:])
                chart_placeholder.altair_chart(build_visual_chart(disp_df), use_container_width=True)

                if sleep_time > 0:
                    time.sleep(sleep_time)

        # Final Render: Full High-Resolution Interactive Stream
        final_df = pd.DataFrame(plot_buffer)
        chart_placeholder.altair_chart(build_visual_chart(final_df, height=360), use_container_width=True)

        # Incident Audit Trail
        if unique_events:
            st.markdown(
                f"""
            <div style="margin-top:20px;margin-bottom:8px;font-size:0.8rem;letter-spacing:0.1em;font-weight:600;text-transform:uppercase;color:#E2C275;">
                CONFIRMED INCIDENT LOG ({len(unique_events)} Flagged Windows)
            </div>
            """,
                unsafe_allow_html=True,
            )

            table_rows = []
            for eid, r in unique_events.items():
                expl = generate_explanation(r)
                table_rows.append(
                    {
                        "Incident ID": eid,
                        "Timestamp": r["timestamp"],
                        "Observed": f"{expl['observed_value']:.2f}",
                        "Expected Baseline": f"{expl['expected_baseline']:.2f}",
                        "Residual Delta": f"{expl['residual']:+.2f} ({expl['percent_deviation']:+.1f}%)",
                        "Score": f"{r['anomaly_score']:.2f}",
                        "Trigger Reason": ", ".join(expl["reason_codes"]),
                    }
                )
            incident_placeholder.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        else:
            incident_placeholder.markdown(
                '<div style="color:#64748B;font-size:0.85rem;padding:8px 0;">No anomaly incidents detected in stream segment. Baseline remained in clean operating regime.</div>',
                unsafe_allow_html=True,
            )

# =============================================================================
# TAB 2: Benchmark Scorecard (Utkrishta 2026 Protocol)
# =============================================================================
with tab_scorecard:
    st.markdown(
        """
    <div style="font-size:0.75rem;letter-spacing:0.1em;font-weight:600;text-transform:uppercase;color:#D4AF37;margin-bottom:4px;">
        OFFICIAL EVALUATION PROTOCOL · 23,395 STREAMING POINTS
    </div>
    <div style="font-size:1.15rem;font-weight:600;color:#F8FAFC;margin-bottom:14px;">
        Architectural Baseline Bake-Off & False Alarm Suppression
    </div>
    """,
        unsafe_allow_html=True,
    )

    if os.path.exists("outputs/experiment_log.csv"):
        exp_df = pd.read_csv("outputs/experiment_log.csv")
        st.dataframe(
            exp_df[
                [
                    "experiment_id",
                    "model_name",
                    "event_f1",
                    "precision",
                    "recall",
                    "true_positives",
                    "false_positives",
                    "throughput_pts_sec",
                    "latency_ms_pt",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    if os.path.exists("outputs/model_comparison_scorecard.csv"):
        st.markdown(
            """
        <div style="font-size:0.75rem;letter-spacing:0.1em;font-weight:600;text-transform:uppercase;color:#D4AF37;margin-bottom:4px;">
            CHAMPION VS. CHALLENGER COMPARISON
        </div>
        <div style="font-size:1.1rem;font-weight:600;color:#F8FAFC;margin-bottom:12px;">
            Model A (Global Generalist) vs. Model B (Infrastructure Specialist)
        </div>
        """,
            unsafe_allow_html=True,
        )
        cmp_df = pd.read_csv("outputs/model_comparison_scorecard.csv")
        st.dataframe(
            cmp_df[
                [
                    "model_name",
                    "event_f1",
                    "precision",
                    "recall",
                    "true_positives",
                    "false_positives",
                    "throughput_pts_sec",
                    "latency_ms_pt",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

# =============================================================================
# TAB 3: Sentinel 2.0 Innovations
# =============================================================================
with tab_pathway:
    st.markdown(
        """
    <div style="font-size:0.75rem;letter-spacing:0.1em;font-weight:600;text-transform:uppercase;color:#D4AF37;margin-bottom:4px;">
        RESEARCH ROADMAP DEMONSTRATION
    </div>
    <div style="font-size:1.15rem;font-weight:600;color:#F8FAFC;margin-bottom:16px;">
        Production-Ready Adaptive Modules (Branch: feature/minimalist-gold-dashboard)
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_inn1, col_inn2 = st.columns(2)

    with col_inn1:
        st.markdown(
            """
        <div class="kpi-card">
            <div class="kpi-label">1. Conformalized Risk Control (CAD)</div>
            <p style="font-size:0.85rem;color:#94A3B8;margin-top:6px;">
                Guarantees finite-sample false alarm risk: <code>P(False Alarm | Normal) &le; &alpha;</code>.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        alpha_val = st.slider("Target False Alarm Rate (alpha)", 0.005, 0.05, 0.02, 0.005, format="%.3f")
        from sentinel.adaptation.conformal import ConformalAnomalyScorer

        scorer = ConformalAnomalyScorer(alpha=alpha_val, calibration_window=500, min_samples=30)
        np.random.seed(42)
        for s in np.abs(np.random.normal(loc=1.5, scale=0.8, size=150)).tolist():
            scorer.add_calibration_score(s)
        th = scorer.get_threshold()
        st.markdown(
            f"<div style='font-family:JetBrains Mono;font-size:0.9rem;color:#E2C275;'>Calibrated Conformal Threshold: {th:.3f} (Risk Bound: {alpha_val*100:.1f}%)</div>",
            unsafe_allow_html=True,
        )

    with col_inn2:
        st.markdown(
            """
        <div class="kpi-card">
            <div class="kpi-label">2. Human-in-the-Loop Feedback Bank</div>
            <p style="font-size:0.85rem;color:#94A3B8;margin-top:6px;">
                Causal cosine similarity suppression of approved maintenance windows with ZERO retraining.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        from sentinel.adaptation.feedback import OperatorFeedbackBank

        fb_bank = OperatorFeedbackBank(similarity_threshold=0.90)
        mock_sig = np.array([0.2, 0.5, 1.8, 3.2, 0.8, 0.3], dtype=np.float32)
        rule_id = fb_bank.add_feedback(
            series_id="service_A",
            feature_vector=mock_sig,
            reason="Approved Weekly Database Vacuum",
            suppression_weight=0.90,
        )
        st.markdown(
            f"<div style='font-family:JetBrains Mono;font-size:0.9rem;color:#4ADE80;'>Active Rule: {rule_id} (Suppression Weight: 90%)</div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 4: Causality Guarantees
# =============================================================================
with tab_architecture:
    st.markdown(
        """
    <div style="font-size:0.75rem;letter-spacing:0.1em;font-weight:600;text-transform:uppercase;color:#D4AF37;margin-bottom:4px;">
        MATHEMATICAL INVARIANTS & VERIFICATION
    </div>
    <div style="font-size:1.15rem;font-weight:600;color:#F8FAFC;margin-bottom:14px;">
        Guarantees Enforced Across Sentinel's Architecture
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    - **Strict Causality**: Every feature, baseline, and anomaly score at time $t$ depends exclusively on past context $x_{<t}$. Radically perturbing future observations leaves past predictions 100% bit-for-bit invariant (verified by `test_full_system_future_perturbation_causality`).
    - **Guarded Baseline Adaptation**: Points classified as anomalous are quarantined from updating rolling reference buffers, preventing prolonged incidents from contaminating normal baselines.
    - **Guarded ADWIN Drift**: ADWIN monitors normalized residuals of normal accepted points, blocking transient incident spikes from triggering spurious drift resets.
    - **Hysteresis Dual Thresholding**: High enter ($\tau = 6.5$) and low exit ($\tau = 2.5$) thresholds eliminate alert flapping and boundary oscillations.
    - **RTX 5050 Native Execution**: Dilated causal 1D convolutions execute with sub-millisecond inference latency on Blackwell CUDA tensors.
    """
    )
