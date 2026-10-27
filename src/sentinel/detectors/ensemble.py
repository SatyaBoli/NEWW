"""Hybrid Ensemble Detector: Calibrated fusion of deep and statistical detectors.

Fuses Causal TCN-AE, Rolling MAD, and EWMA into a calibrated multi-perspective detector.
"""
from __future__ import annotations

import math
from typing import Optional

from ..contracts import Point, RawDetectorOutput
from .base import BaseDetector
from .ewma import EWMADetector
from .rolling_mad import RollingMedianMADDetector
from .tcn_autoencoder import CausalTCNAEDetector


class HybridEnsembleDetector(BaseDetector):
    """Calibrated ensemble combining statistical baselines and deep neural reconstruction."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "models/sentinel_specialist_best.pt",
        weights: tuple[float, float, float] = (0.45, 0.25, 0.30),  # (MAD, EWMA, TCN-AE)
        scale_floor: float = 0.05,
        mad_window: int = 50,
        ewma_alpha: float = 0.15,
        tcn_window: int = 32,
        version: str = "ensemble-0.2.0",
    ) -> None:
        self.weights = weights
        self._version = version
        self.scale_floor = scale_floor

        self.mad_detector = RollingMedianMADDetector(window_size=mad_window, warmup_steps=30, scale_floor=scale_floor)
        self.ewma_detector = EWMADetector(alpha=ewma_alpha, warmup_steps=30, scale_floor=scale_floor)
        self.tcn_detector = CausalTCNAEDetector(
            window_size=tcn_window,
            warmup_steps=25,
            scale_floor=scale_floor,
            checkpoint_path=checkpoint_path,
        )

    @property
    def version(self) -> str:
        return self._version

    def score_one(self, point: Point) -> RawDetectorOutput:
        out_mad = self.mad_detector.score_one(point)
        out_ewma = self.ewma_detector.score_one(point)
        out_tcn = self.tcn_detector.score_one(point)

        w_mad, w_ewma, w_tcn = self.weights

        # Calibrated weighted combination
        combined_score = (
            w_mad * out_mad.raw_score +
            w_ewma * out_ewma.raw_score +
            w_tcn * out_tcn.raw_score
        )

        # Consensus baseline
        consensus_baseline = (
            w_mad * out_mad.baseline +
            w_ewma * out_ewma.baseline +
            w_tcn * out_tcn.baseline
        )
        combined_residual = point.value - consensus_baseline

        is_warmup = (
            out_mad.context.get("is_warmup", False) or
            out_tcn.context.get("is_warmup", False)
        )

        contributors = []
        if out_mad.raw_score > 3.0:
            contributors.append("statistical_spike_mad")
        if out_ewma.raw_score > 3.0:
            contributors.append("trend_residual_ewma")
        if out_tcn.raw_score > 3.0:
            contributors.append("temporal_pattern_distortion_tcn")
        if not contributors:
            contributors.append("nominal_concurrence")

        return RawDetectorOutput(
            raw_score=float(combined_score),
            observed=float(point.value),
            baseline=float(consensus_baseline),
            residual=float(combined_residual),
            context={
                "mad_score": float(out_mad.raw_score),
                "ewma_score": float(out_ewma.raw_score),
                "tcn_score": float(out_tcn.raw_score),
                "is_warmup": is_warmup,
                "contributors": contributors,
            },
        )

    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        self.mad_detector.update_state(point, is_anomalous=is_anomalous)
        self.ewma_detector.update_state(point, is_anomalous=is_anomalous)
        self.tcn_detector.update_state(point, is_anomalous=is_anomalous)

    def reset_state(self, series_id: Optional[str] = None) -> None:
        self.mad_detector.reset_state(series_id)
        self.ewma_detector.reset_state(series_id)
        self.tcn_detector.reset_state(series_id)
