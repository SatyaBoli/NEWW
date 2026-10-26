"""Unit tests for the event-based evaluation metric harness."""
from __future__ import annotations

import pytest

from sentinel.evaluation.event_f1 import evaluate_event_f1


def test_perfect_detection():
    gt = {"s1": [["2026-01-01T00:10:00", "2026-01-01T00:20:00"]]}
    preds = [
        {"series_id": "s1", "timestamp": "2026-01-01T00:05:00", "label": "NORMAL", "event_id": None},
        {"series_id": "s1", "timestamp": "2026-01-01T00:12:00", "label": "ANOMALOUS", "event_id": "s1_evt_0001"},
        {"series_id": "s1", "timestamp": "2026-01-01T00:15:00", "label": "ANOMALOUS", "event_id": "s1_evt_0001"},
        {"series_id": "s1", "timestamp": "2026-01-01T00:25:00", "label": "NORMAL", "event_id": None},
    ]

    m = evaluate_event_f1(preds, gt)
    assert m.true_positives == 1
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.event_f1 == 1.0
    # Delay from 00:10:00 to 00:12:00 = 120 seconds
    assert m.mean_detection_delay_seconds == 120.0


def test_false_alarm_and_miss():
    gt = {
        "s1": [
            ["2026-01-01T00:10:00", "2026-01-01T00:20:00"],  # will be missed
            ["2026-01-01T01:00:00", "2026-01-01T01:10:00"],  # will be detected
        ]
    }
    preds = [
        # Spurious detection at 00:30 (False Positive)
        {"series_id": "s1", "timestamp": "2026-01-01T00:30:00", "label": "ANOMALOUS", "event_id": "s1_evt_0001"},
        # Real detection at 01:02 (True Positive)
        {"series_id": "s1", "timestamp": "2026-01-01T01:02:00", "label": "ANOMALOUS", "event_id": "s1_evt_0002"},
    ]

    m = evaluate_event_f1(preds, gt)
    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.false_negatives == 1
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.event_f1 == 0.5


def test_cooldown_merged_event_counted_once():
    """An event broken by a small gap that was merged under cooldown should count as 1 event."""
    gt = {"s1": [["2026-01-01T00:10:00", "2026-01-01T00:30:00"]]}
    preds = [
        {"series_id": "s1", "timestamp": "2026-01-01T00:12:00", "label": "ANOMALOUS", "event_id": "s1_evt_0001"},
        {"series_id": "s1", "timestamp": "2026-01-01T00:13:00", "label": "NORMAL", "event_id": None},
        {"series_id": "s1", "timestamp": "2026-01-01T00:14:00", "label": "ANOMALOUS", "event_id": "s1_evt_0001"},
    ]

    m = evaluate_event_f1(preds, gt)
    assert m.true_positives == 1
    assert m.false_positives == 0
    assert m.event_f1 == 1.0
