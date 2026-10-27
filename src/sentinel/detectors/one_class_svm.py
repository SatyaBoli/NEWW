"""Statistical Baseline Detector: Causal Streaming One-Class Support Vector Machine (One-Class SVM).

Learns a non-linear decision boundary around normal streaming state in RKHS.
Satisfies Challenge 2 specification: Permitted Statistical Baselines (One-Class SVM).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
from sklearn.svm import OneClassSVM

from ..contracts import Point, RawDetectorOutput
from ..features.causal_features import CausalFeatureExtractor
from .base import BaseDetector


class StreamingOneClassSVMDetector(BaseDetector):
    """Causal sliding-window One-Class SVM detector."""

    def __init__(
        self,
        window_size: int = 80,
        warmup_steps: int = 35,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        retrain_interval: int = 200,
        scale_floor: float = 0.05,
        version: str = "ocsvm-0.1.0",
    ) -> None:
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.retrain_interval = retrain_interval
        self.scale_floor = scale_floor
        self._version = version

        self._feature_extractor = CausalFeatureExtractor(window_size=20, warmup_steps=12)
        self._histories: dict[str, deque[np.ndarray]] = {}
        self._models: dict[str, OneClassSVM] = {}
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
        model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
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
            # In OneClassSVM, decision_function is signed distance to separating hyperplane.
            # Positive for normal points, negative for anomalies.
            raw_score = max(0.0, -df_score * 6.0)
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
                "ocsvm_margin": df_score if 'df_score' in locals() else 0.0,
                "is_warmup": False,
                "n_samples": len(hist),
                "contributors": ["rbf_kernel_distance", "support_vector_margin"],
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
