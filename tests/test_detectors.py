"""Unit tests for Sentinel detectors verifying causality, edge cases, and isolation."""
from __future__ import annotations

import math
import numpy as np
import pytest

from sentinel.contracts import Point
from sentinel.detectors.ewma import EWMADetector
from sentinel.detectors.rolling_mad import RollingMedianMADDetector


def test_mad_causality():
    """Future changes must not alter past scores."""
    det = RollingMedianMADDetector(window_size=20, warmup_steps=10)

    # Feed initial 30 points
    series = [float(i % 10) for i in range(30)]
    scores_initial = []
    for i, v in enumerate(series):
        p = Point("s1", f"2026-01-01T00:{i:02d}:00", v)
        out = det.score_one(p)
        det.update_state(p, is_anomalous=False)
        scores_initial.append(out.raw_score)

    # Re-run same prefix with a radically different point inserted at end
    det.reset_state()
    scores_rerun = []
    for i, v in enumerate(series):
        p = Point("s1", f"2026-01-01T00:{i:02d}:00", v)
        out = det.score_one(p)
        det.update_state(p, is_anomalous=False)
        scores_rerun.append(out.raw_score)

    assert scores_initial == scores_rerun


def test_mad_flatline_zero_variance():
    """Constant flatline series (MAD=0) must not crash or produce NaN/inf."""
    det = RollingMedianMADDetector(window_size=20, warmup_steps=5, scale_floor=1e-4)

    for i in range(25):
        p = Point("s_flat", f"2026-01-01T00:{i:02d}:00", 42.0)
        out = det.score_one(p)
        det.update_state(p, is_anomalous=False)
        assert math.isfinite(out.raw_score)
        assert out.raw_score >= 0.0

    # Inject sudden departure
    p_spike = Point("s_flat", "2026-01-01T00:26:00", 50.0)
    out_spike = det.score_one(p_spike)
    assert out_spike.raw_score > 10.0
    assert math.isfinite(out_spike.raw_score)


def test_ewma_spike_detection():
    det = EWMADetector(alpha=0.2, warmup_steps=5)

    # Normal baseline around 100
    for i in range(20):
        p = Point("s1", f"2026-01-01T00:{i:02d}:00", 100.0 + (i % 2))
        det.score_one(p)
        det.update_state(p, is_anomalous=False)

    # Outlier spike
    spike = Point("s1", "2026-01-01T00:21:00", 250.0)
    out = det.score_one(spike)
    assert out.raw_score > 5.0
    assert out.residual > 100.0


def test_interleaved_series_isolation():
    """Alternating points between two series must yield identical results to sequential."""
    det_seq = RollingMedianMADDetector(window_size=15, warmup_steps=5)
    det_inter = RollingMedianMADDetector(window_size=15, warmup_steps=5)

    pts_a = [Point("A", f"2026-01-01T00:{i:02d}:00", 10.0 + (i % 5)) for i in range(20)]
    pts_b = [Point("B", f"2026-01-01T00:{i:02d}:00", 500.0 + (i % 3) * 50) for i in range(20)]

    # Sequential run for A
    scores_a_seq = []
    for p in pts_a:
        out = det_seq.score_one(p)
        det_seq.update_state(p, is_anomalous=False)
        scores_a_seq.append(out.raw_score)

    # Interleaved run A and B
    scores_a_inter = []
    for pa, pb in zip(pts_a, pts_b):
        out_a = det_inter.score_one(pa)
        det_inter.update_state(pa, is_anomalous=False)
        scores_a_inter.append(out_a.raw_score)

        out_b = det_inter.score_one(pb)
        det_inter.update_state(pb, is_anomalous=False)

    assert np.allclose(scores_a_seq, scores_a_inter)
