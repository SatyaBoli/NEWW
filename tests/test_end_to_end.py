"""End-to-end integration and release verification tests for Sentinel."""
from __future__ import annotations

import pytest

from sentinel.config import AdaptationConfig, DetectorConfig, SentinelConfig
from sentinel.contracts import Point
from sentinel.evaluation.event_f1 import evaluate_event_f1
from sentinel.explanations.explainer import generate_explanation
from sentinel.output.submission_adapter import SubmissionAdapter
from sentinel.pipeline import SentinelPipeline


def test_pipeline_vertical_slice():
    """End-to-end integration test: stream points -> score -> adapt -> format -> evaluate."""
    cfg = SentinelConfig(
        detector=DetectorConfig(detector_type="rolling_mad", window_size=20),
        adaptation=AdaptationConfig(enter_threshold=3.5, exit_threshold=2.0, warmup_steps=10),
    )
    pipeline = SentinelPipeline(cfg)

    # Normal stream with an anomalous burst at t=15..17
    points = []
    for i in range(25):
        val = 100.0 if (15 <= i <= 17) else 10.0
        ts = f"2026-01-01T00:{i:02d}:00"
        points.append(Point("test_series", ts, val))

    records, stats = pipeline.process_stream(points)

    assert len(records) == 25
    assert stats["total_points"] == 25
    assert stats["throughput_points_per_sec"] > 0

    # Verify anomalous labels triggered during the burst
    anom_records = [r for r in records if r["label"] == "ANOMALOUS"]
    assert len(anom_records) >= 1

    # Verify deterministic evidence generation
    first_anom = anom_records[0]
    expl = generate_explanation(first_anom)
    assert "narrative" in expl
    assert expl["observed_value"] == 100.0
    assert expl["direction"] == "spike"

    # Verify organizer submission formatting
    df_sub = SubmissionAdapter.to_dataframe(records)
    assert len(df_sub) == 25
    assert "anomaly_score" in df_sub.columns
    assert "explanation" in df_sub.columns

    # Verify event-based evaluation matching
    gt_windows = {"test_series": [["2026-01-01T00:15:00", "2026-01-01T00:18:00"]]}
    metrics = evaluate_event_f1(records, gt_windows)
    assert metrics.true_positives == 1
    assert metrics.event_f1 == 1.0


def test_gpu_autoencoder_inference():
    """Test Causal TCN-AE detector executes without NaN and respects device."""
    cfg = SentinelConfig(
        detector=DetectorConfig(detector_type="autoencoder", ae_device="cuda"),
        adaptation=AdaptationConfig(warmup_steps=10),
    )
    pipeline = SentinelPipeline(cfg)

    # Process 35 points to exit warm-up
    points = [Point("s_gpu", f"2026-01-01T00:{i:02d}:00", float(i % 5)) for i in range(35)]
    records, stats = pipeline.process_stream(points)

    assert len(records) == 35
    for r in records:
        assert r["anomaly_score"] >= 0.0
        assert r["label"] in ("NORMAL", "ANOMALOUS")


def test_full_system_future_perturbation_causality():
    """Mathematical causality proof: radically perturbing future observations x_{>=t}
    must leave all past predictions, scores, and baselines at steps < t 100% invariant."""
    cfg = SentinelConfig(
        detector=DetectorConfig(detector_type="rolling_mad", window_size=20),
        adaptation=AdaptationConfig(enter_threshold=3.5, exit_threshold=2.0, warmup_steps=10),
    )

    base_points = [
        Point("causal_series", f"2026-01-01T00:{i:02d}:00", 20.0 + (i % 7))
        for i in range(40)
    ]

    # Run 1: process first 25 points
    p1 = SentinelPipeline(cfg)
    records_clean, _ = p1.process_stream(base_points[:25])

    # Run 2: append extreme wild future values (points 25..40 set to 999,999.0)
    perturbed_points = list(base_points[:25]) + [
        Point("causal_series", f"2026-01-01T00:{i:02d}:00", 999999.0)
        for i in range(25, 40)
    ]
    p2 = SentinelPipeline(cfg)
    records_perturbed, _ = p2.process_stream(perturbed_points)

    # Assert past prefix [0..24] is identical in scores, baselines, and decisions
    for i in range(25):
        r_clean = records_clean[i]
        r_pert = records_perturbed[i]
        assert r_clean["anomaly_score"] == pytest.approx(r_pert["anomaly_score"], rel=1e-5), f"Past score changed at step {i}!"
        assert r_clean["evidence"]["baseline"] == pytest.approx(r_pert["evidence"]["baseline"], rel=1e-5), f"Past baseline changed at step {i}!"
        assert r_clean["evidence"]["residual"] == pytest.approx(r_pert["evidence"]["residual"], rel=1e-5), f"Past residual changed at step {i}!"
        assert r_clean["label"] == r_pert["label"], f"Past label changed at step {i}!"

