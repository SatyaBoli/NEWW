"""Conformalized Anomaly Detection (CAD) with distribution-free risk guarantees.

Implements inductive and streaming causal conformal prediction. Given an operator-specified
significance level alpha (e.g. 0.01 for 1% false alarm tolerance), computes dynamic
non-conformity quantiles and empirical p-values with theoretical coverage bounds:
    P(False Alarm | Normal) <= alpha.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np


class ConformalAnomalyScorer:
    """Distribution-free conformal risk controller for streaming non-conformity scores."""

    def __init__(
        self,
        alpha: float = 0.02,
        calibration_window: int = 500,
        min_samples: int = 50,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.calibration_window = calibration_window
        self.min_samples = min_samples

        self._cal_scores: deque[float] = deque(maxlen=calibration_window)
        self._sorted_cache: Optional[np.ndarray] = None
        self._threshold_cache: Optional[float] = None

    @property
    def sample_count(self) -> int:
        return len(self._cal_scores)

    @property
    def is_calibrated(self) -> bool:
        return len(self._cal_scores) >= self.min_samples

    def add_calibration_score(self, score: float) -> None:
        """Ingest certified normal non-conformity score into calibration buffer."""
        if math.isfinite(score):
            self._cal_scores.append(float(score))
            self._sorted_cache = None
            self._threshold_cache = None

    def get_threshold(self) -> float:
        """Compute conformal non-conformity threshold q_hat at (1 - alpha) level."""
        if not self.is_calibrated:
            return float("inf")

        if self._threshold_cache is not None:
            return self._threshold_cache

        if self._sorted_cache is None:
            self._sorted_cache = np.sort(np.array(self._cal_scores, dtype=np.float64))

        n = len(self._sorted_cache)
        # Standard finite-sample conformal quantile index: ceil((n + 1) * (1 - alpha)) / n
        q_idx = int(math.ceil((n + 1) * (1.0 - self.alpha))) - 1
        q_idx = max(0, min(q_idx, n - 1))
        threshold = float(self._sorted_cache[q_idx])
        self._threshold_cache = threshold
        return threshold

    def compute_p_value(self, test_score: float) -> float:
        """Compute empirical conformal p-value: (1 + sum(S_i >= test_score)) / (n + 1)."""
        if not math.isfinite(test_score):
            return 0.0

        n = len(self._cal_scores)
        if n < self.min_samples:
            return 1.0  # Conservative during calibration

        arr = np.array(self._cal_scores, dtype=np.float64)
        count_ge = int(np.sum(arr >= test_score))
        p_val = (1.0 + count_ge) / (n + 1.0)
        return float(min(max(p_val, 0.0), 1.0))

    def evaluate(self, test_score: float) -> tuple[bool, float, float]:
        """Evaluate observation.

        Returns:
            (is_conformal_anomaly, p_value, conformal_threshold)
        """
        p_val = self.compute_p_value(test_score)
        threshold = self.get_threshold()
        is_anomaly = bool(p_val < self.alpha and self.is_calibrated)
        return is_anomaly, p_val, threshold

    def reset(self) -> None:
        self._cal_scores.clear()
        self._sorted_cache = None
        self._threshold_cache = None
