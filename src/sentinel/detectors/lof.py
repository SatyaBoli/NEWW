"""Statistical Baseline Detector: Causal Streaming Local Outlier Factor (LOF).

Measures local density deviation of causal streaming lag and rolling features.
Satisfies Challenge 2 specification: Permitted Statistical Baselines (LOF).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ..contracts import Point, RawDetectorOutput
from ..features.causal_features import CausalFeatureExtractor
from .base import BaseDetector


class StreamingLOFDetector(BaseDetector):
    """Causal sliding-window Local Outlier Factor detector."""

    def __init__(
        self,
        window_size: int = 80,
        warmup_steps: int = 35,
        n_neighbors: int = 20,
        contamination: float = 0.02,
        retrain_interval: int = 150,
        scale_floor: float = 0.05,
        version: str = "lof-0.1.0",
    ) -> None:
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.retrain_interval = retrain_interval
        self.scale_floor = scale_floor
        self._version = version

        self._feature_extractor = CausalFeatureExtractor(window_size=20, warmup_steps=12)
        self._histories: dict[str, deque[np.ndarray]] = {}
        self._models: dict[str, LocalOutlierFactor] = {}
        self._steps_since_train: dict[str, int] = {}

    @property
    def version(self) -> str:
        return self._version

    def _get_history(self, series_id: str) -> deque[np.ndarray]:
        hist = self._histories.get(series_id)
        if hist is None:
            hist = self._histories[series_id] = deque(maxlen=self.window_size)
        return hist

    def _train_model(self, series_id: str) -> None:
        hist = self._get_history(series_id)
        if len(hist) < self.warmup_steps:
            return
        X = np.vstack(list(hist))
        k = min(self.n_neighbors, len(hist) - 1)
        if k < 2:
            return
        model = LocalOutlierFactor(
            n_neighbors=k,
            contamination=self.contamination,
            novelty=True,
            n_jobs=1,
        )
        try:
            model.fit(X)
            self._models[series_id] = model
            self._steps_since_train[series_id] = 0
        except Exception:
            pass

    def score_one(self, point: Point) -> RawDetectorOutput:
        feat = self._feature_extractor.extract(point)
        x_vec = feat.to_array().reshape(1, -1)
        hist = self._get_history(point.series_id)

        model = self._models.get(point.series_id)
        if model is None or len(hist) < self.warmup_steps:
            raw_score = abs(feat.z_score)
            return RawDetectorOutput(
                raw_score=float(raw_score),
                observed=float(point.value),
                baseline=float(feat.rolling_mean),
                residual=float(point.value - feat.rolling_mean),
                context={"is_warmup": True, "model": "fallback_z", "n_samples": len(hist)},
            )

        try:
            df_score = float(model.decision_function(x_vec)[0])
            # In LOF with novelty=True: decision_function produces negative values for anomalies
            # and positive values for normal points.
            raw_score = max(0.0, -df_score * 8.0)
        except Exception:
            raw_score = abs(feat.z_score)

        if not math.isfinite(raw_score):
            raw_score = 0.0

        diff = point.value - feat.rolling_mean
        return RawDetectorOutput(
            raw_score=float(raw_score),
            observed=float(point.value),
            baseline=float(feat.rolling_mean),
            residual=float(diff),
            context={
                "lof_score": df_score if 'df_score' in locals() else 0.0,
                "is_warmup": False,
                "n_samples": len(hist),
                "contributors": ["local_reachability_density", "k_nearest_neighbors"],
            },
        )

    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        feat = self._feature_extractor.extract(point)
        self._feature_extractor.update(point)

        hist = self._get_history(point.series_id)
        if not is_anomalous or len(hist) < self.warmup_steps:
            hist.append(feat.to_array())

        steps = self._steps_since_train.get(point.series_id, 0) + 1
        self._steps_since_train[point.series_id] = steps

        if (point.series_id not in self._models and len(hist) >= self.warmup_steps) or (
            steps >= self.retrain_interval and len(hist) >= self.warmup_steps
        ):
            self._train_model(point.series_id)

    def reset_state(self, series_id: Optional[str] = None) -> None:
        self._feature_extractor.reset(series_id)
        if series_id:
            self._histories.pop(series_id, None)
            self._models.pop(series_id, None)
            self._steps_since_train.pop(series_id, None)
        else:
            self._histories.clear()
            self._models.clear()
            self._steps_since_train.clear()
