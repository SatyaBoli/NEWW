"""Baseline Detector B0: Causal Rolling Median / MAD with robust scale floor.

Implements Iglewicz & Hoaglin (1993) modified z-score outlier detection:
  Score = 0.6745 * |x - median| / MAD
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from ..contracts import Point, RawDetectorOutput
from .base import BaseDetector


class RollingMedianMADDetector(BaseDetector):
    """Causal rolling window median and MAD anomaly detector."""

    def __init__(
        self,
        window_size: int = 50,
        warmup_steps: int = 30,
        scale_floor: float = 1e-4,
        version: str = "rolling_mad-0.1.0",
    ) -> None:
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.scale_floor = scale_floor
        self._version = version
        self._buffers: dict[str, deque[float]] = {}

    @property
    def version(self) -> str:
        return self._version

    def _get_buffer(self, series_id: str) -> deque[float]:
        buf = self._buffers.get(series_id)
        if buf is None:
            buf = self._buffers[series_id] = deque(maxlen=self.window_size)
        return buf

    def score_one(self, point: Point) -> RawDetectorOutput:
        buf = self._get_buffer(point.series_id)
        n = len(buf)

        if n < 3:
            # Cold start: insufficient history to establish distribution
            return RawDetectorOutput(
                raw_score=0.0,
                observed=point.value,
                baseline=point.value,
                residual=0.0,
                context={"median": point.value, "mad": 0.0, "is_warmup": True, "window_size": n},
            )

        vals = np.array(buf, dtype=np.float64)
        med = float(np.median(vals))
        abs_diff = np.abs(vals - med)
        mad = float(np.median(abs_diff))
        mad_eff = max(mad, self.scale_floor)

        diff = point.value - med
        # Iglewicz & Hoaglin modified z-score
        score = 0.6745 * abs(diff) / mad_eff

        if not math.isfinite(score):
            score = 1e6

        is_warmup = n < self.warmup_steps

        return RawDetectorOutput(
            raw_score=float(score),
            observed=float(point.value),
            baseline=float(med),
            residual=float(diff),
            context={
                "median": med,
                "mad": mad,
                "mad_effective": mad_eff,
                "is_warmup": is_warmup,
                "window_size": n,
                "contributors": ["median_deviation"],
            },
        )

    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        buf = self._get_buffer(point.series_id)
        # Guarded update: do not push into baseline if marked anomalous,
        # unless still within the initial warm-up period
        if not is_anomalous or len(buf) < self.warmup_steps:
            if math.isfinite(point.value):
                buf.append(point.value)

    def reset_state(self, series_id: Optional[str] = None) -> None:
        if series_id:
            if series_id in self._buffers:
                self._buffers[series_id].clear()
        else:
            self._buffers.clear()
