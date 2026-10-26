"""Causal Periodicity Discovery via auto-correlation peak detection.

Automatically infers dominant seasonal rhythms (diurnal, shift-based, or cyclical)
from streaming warm-up buffers without requiring manual period specification.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


class CausalPeriodicityDetector:
    """Estimates dominant streaming periodicity using causal auto-correlation analysis."""

    def __init__(
        self,
        min_period: int = 12,
        max_period: int = 400,
        prominence_threshold: float = 0.30,
    ) -> None:
        self.min_period = min_period
        self.max_period = max_period
        self.prominence_threshold = prominence_threshold

    def discover_period(self, series_buffer: Sequence[float]) -> tuple[Optional[int], float]:
        """Infer dominant cycle length and confidence from observed past buffer.

        Returns:
            (dominant_period, confidence) or (None, 0.0) if no rhythm detected.
        """
        arr = np.array(series_buffer, dtype=np.float64)
        n = len(arr)
        if n < self.min_period * 3:
            return None, 0.0

        # Detrend via first-difference for stationary ACF
        diff = np.diff(arr)
        var = np.var(diff)
        if var < 1e-8:
            return None, 0.0

        diff_norm = (diff - np.mean(diff)) / (np.std(diff) + 1e-8)
        max_lag = min(self.max_period, (n // 2) - 1)

        if max_lag <= self.min_period:
            return None, 0.0

        # Compute autocorrelation for candidate lags
        lags = np.arange(self.min_period, max_lag)
        acf = np.zeros(len(lags), dtype=np.float64)

        for idx, lag in enumerate(lags):
            # Causal overlap
            x1 = diff_norm[:-lag]
            x2 = diff_norm[lag:]
            acf[idx] = np.mean(x1 * x2)

        if len(acf) == 0:
            return None, 0.0

        # Find the first prominent local peak (fundamental frequency)
        for i in range(1, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]:
                if acf[i] >= self.prominence_threshold:
                    left_idx = max(0, i - 5)
                    right_idx = min(len(acf) - 1, i + 5)
                    local_baseline = float((acf[left_idx] + acf[right_idx]) / 2.0)
                    prominence = acf[i] - local_baseline
                    if prominence >= 0.10:
                        fundamental_lag = int(lags[i])
                        confidence = float(min(max(acf[i], 0.0), 1.0))
                        return fundamental_lag, confidence

        # Fallback to global max if no distinct local peak passed strict test
        best_idx = int(np.argmax(acf))
        best_corr = float(acf[best_idx])
        if best_corr >= self.prominence_threshold:
            return int(lags[best_idx]), float(min(max(best_corr, 0.0), 1.0))

        return None, 0.0
