"""Unit tests for the Event State Machine."""
from __future__ import annotations

import pytest

from sentinel.config import AdaptationConfig
from sentinel.contracts import LABEL_ANOMALOUS, LABEL_NORMAL
from sentinel.events.state_machine import (
    COOLDOWN_MERGE,
    EVENT_START,
    HYSTERESIS_HOLD,
    NON_FINITE_SCORE,
    WARMUP,
    EventStateMachine,
)


def test_warmup_blocks_events():
    cfg = AdaptationConfig(warmup_steps=5, enter_threshold=3.5, exit_threshold=2.0)
    sm = EventStateMachine("s1", cfg)

    # During warmup, even huge scores should remain LABEL_NORMAL
    for _ in range(5):
        dec = sm.step(10.0)
        assert dec.label == LABEL_NORMAL
        assert WARMUP in dec.reason_codes
        assert dec.event_id is None

    # After warmup, score >= enter_threshold opens event
    dec = sm.step(4.0)
    assert dec.label == LABEL_ANOMALOUS
    assert EVENT_START in dec.reason_codes
    assert dec.event_id is not None


def test_hysteresis_behavior():
    cfg = AdaptationConfig(warmup_steps=0, enter_threshold=3.5, exit_threshold=2.0)
    sm = EventStateMachine("s1", cfg)

    # Trigger event
    d1 = sm.step(4.0)
    assert d1.label == LABEL_ANOMALOUS
    assert d1.event_id is not None
    evt_id = d1.event_id

    # Score in borderline zone [2.0, 3.5) holds event open via hysteresis
    d2 = sm.step(2.5)
    assert d2.label == LABEL_ANOMALOUS
    assert d2.event_id == evt_id
    assert HYSTERESIS_HOLD in d2.reason_codes

    # Score drops below exit_threshold (2.0) closes event
    d3 = sm.step(1.5)
    assert d3.label == LABEL_NORMAL
    assert d3.event_id is None


def test_cooldown_event_merge():
    cfg = AdaptationConfig(warmup_steps=0, enter_threshold=3.5, exit_threshold=2.0, cooldown_steps=2)
    sm = EventStateMachine("s1", cfg)

    # Open event 1
    d1 = sm.step(4.0)
    assert d1.label == LABEL_ANOMALOUS
    evt_id = d1.event_id

    # Close event
    d2 = sm.step(1.0)
    assert d2.label == LABEL_NORMAL

    # Re-entry within cooldown_steps should reuse same event_id
    d3 = sm.step(4.5)
    assert d3.label == LABEL_ANOMALOUS
    assert d3.event_id == evt_id
    assert COOLDOWN_MERGE in d3.reason_codes


def test_non_finite_scores():
    cfg = AdaptationConfig(warmup_steps=0, max_score=100.0)
    sm = EventStateMachine("s1", cfg)

    dec_nan = sm.step(float("nan"))
    assert NON_FINITE_SCORE in dec_nan.reason_codes
    assert dec_nan.anomaly_score == 100.0

    dec_inf = sm.step(float("inf"))
    assert NON_FINITE_SCORE in dec_inf.reason_codes
    assert dec_inf.anomaly_score == 100.0
