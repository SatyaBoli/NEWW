"""Unit tests for Sentinel Next-Gen Pathway modules.

Tests:
1. Conformal Anomaly Scorer (Distribution-free statistical risk guarantees)
2. Operator Feedback Bank (Human-in-the-Loop spurious alarm suppression)
3. Causal Periodicity Discovery (Dominant seasonal cycle discovery)
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from sentinel.adaptation.conformal import ConformalAnomalyScorer
from sentinel.adaptation.feedback import OperatorFeedbackBank
from sentinel.features.periodicity import CausalPeriodicityDetector


def test_conformal_scorer_coverage():
    scorer = ConformalAnomalyScorer(alpha=0.05, calibration_window=200, min_samples=30)
    assert not scorer.is_calibrated

    # Feed normal scores from N(0, 1) absolute distribution
    np.random.seed(42)
    normal_scores = np.abs(np.random.randn(100)).tolist()
    for s in normal_scores:
        scorer.add_calibration_score(s)

    assert scorer.is_calibrated
    thresh = scorer.get_threshold()
    assert math.isfinite(thresh)
    assert thresh > 1.0

    # Test an extreme outlier (score = 15.0)
    is_anom, p_val, applied_th = scorer.evaluate(15.0)
    assert is_anom is True
    assert p_val < 0.05
    assert applied_th == thresh

    # Test a typical normal score (score = 0.2)
    is_anom_norm, p_val_norm, _ = scorer.evaluate(0.2)
    assert is_anom_norm is False
    assert p_val_norm > 0.05


def test_operator_feedback_bank():
    bank = OperatorFeedbackBank(similarity_threshold=0.90)

    # Simulated feature vector of a known benign deployment spike
    pattern = np.array([0.1, 0.2, 0.8, 1.5, 0.4, 0.1], dtype=np.float32)
    rule_id = bank.add_feedback(
        series_id="service_cpu_01",
        feature_vector=pattern,
        reason="Approved weekly CI/CD benchmark run",
        suppression_weight=0.90,
    )
    assert rule_id.startswith("FB_service_cpu_01")
    assert len(bank.list_rules("service_cpu_01")) == 1

    # Exact matching vector arrives
    suppressed, discount, matched_rule = bank.check_suppression("service_cpu_01", pattern)
    assert suppressed is True
    assert discount <= 0.15
    assert matched_rule == rule_id

    # Unrelated vector arrives (e.g. flatline or inverted spike)
    unrelated_pattern = np.array([-1.0, -0.8, -0.2, 0.0, 0.1, 0.1], dtype=np.float32)
    suppressed2, discount2, _ = bank.check_suppression("service_cpu_01", unrelated_pattern)
    assert suppressed2 is False
    assert discount2 == 1.0


def test_periodicity_discovery():
    detector = CausalPeriodicityDetector(min_period=12, max_period=100, prominence_threshold=0.30)

    # Generate synthetic series with a strict 48-step sinusoidal period
    t = np.arange(300)
    period_true = 48
    signal = 10.0 + 5.0 * np.sin(2 * np.pi * t / period_true) + np.random.normal(0, 0.1, len(t))

    discovered_period, confidence = detector.discover_period(signal)
    assert discovered_period is not None
    # Estimated period should be within 1 step of true period 48
    assert abs(discovered_period - period_true) <= 1
    assert confidence > 0.40
