"""Unified Sentinel streaming anomaly detection pipeline.

Connects data streaming, causal detectors, adaptation layer, and evidence generation.
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterable, Iterator, Optional, Sequence

from .adaptation.guarded_update import AdaptationLayer
from .config import SentinelConfig
from .contracts import Point, PredictionRecord
from .detectors import (
    BaseDetector,
    CausalTCNAEDetector,
    EWMADetector,
    HybridEnsembleDetector,
    RollingMedianMADDetector,
    StreamingIsolationForestDetector,
)


class SentinelPipeline:
    """Master streaming anomaly detection pipeline."""

    def __init__(self, config: Optional[SentinelConfig] = None) -> None:
        self.config = config or SentinelConfig()
        self.detector = self._build_detector(self.config)
        self.layer = AdaptationLayer(detector=self.detector, config=self.config.adaptation)

    @classmethod
    def from_yaml(cls, path: str) -> "SentinelPipeline":
        cfg = SentinelConfig.from_yaml(path)
        return cls(cfg)

    def _build_detector(self, cfg: SentinelConfig) -> BaseDetector:
        d_type = cfg.detector.detector_type.lower()
        if d_type in ("rolling_mad", "mad", "baseline"):
            return RollingMedianMADDetector(
                window_size=cfg.detector.window_size,
                warmup_steps=cfg.adaptation.warmup_steps,
                scale_floor=cfg.detector.scale_floor,
            )
        elif d_type == "ewma":
            return EWMADetector(
                alpha=cfg.detector.ewma_alpha,
                warmup_steps=cfg.adaptation.warmup_steps,
                scale_floor=cfg.detector.scale_floor,
            )
        elif d_type in ("iforest", "isolation_forest"):
            return StreamingIsolationForestDetector(
                window_size=cfg.detector.window_size,
                n_estimators=cfg.detector.iforest_n_estimators,
                contamination=cfg.detector.iforest_contamination,
            )
        elif d_type in ("autoencoder", "tcn", "tcn_ae"):
            chk = cfg.detector.checkpoint_path
            if not chk:
                for cand in ("models/sentinel_specialist_best.pt", "models/sentinel_tcn_best.pt"):
                    if os.path.exists(cand):
                        chk = cand
                        break
            return CausalTCNAEDetector(
                window_size=cfg.detector.window_size if cfg.detector.window_size <= 64 else 32,
                device=cfg.detector.ae_device,
                scale_floor=cfg.detector.scale_floor,
                checkpoint_path=chk,
            )
        elif d_type in ("ensemble", "hybrid"):
            chk = cfg.detector.checkpoint_path
            if not chk:
                for cand in ("models/sentinel_specialist_best.pt", "models/sentinel_tcn_best.pt"):
                    if os.path.exists(cand):
                        chk = cand
                        break
            return HybridEnsembleDetector(
                checkpoint_path=chk,
                scale_floor=cfg.detector.scale_floor,
                mad_window=cfg.detector.window_size,
                ewma_alpha=cfg.detector.ewma_alpha,
            )
        else:
            raise ValueError(f"Unknown detector type: {d_type}")

    def process_point(self, point: Point) -> PredictionRecord:
        """Process a single streaming point in strictly causal order."""
        return self.layer.process(point)

    def process_stream(
        self, stream: Iterable[Point]
    ) -> tuple[list[PredictionRecord], dict[str, float]]:
        """Stream an iterable of points, collecting results and profiling runtime speed."""
        records: list[PredictionRecord] = []
        start_time = time.perf_counter()

        for pt in stream:
            rec = self.layer.process(pt)
            records.append(rec)

        total_time = time.perf_counter() - start_time
        n_points = len(records)
        points_per_sec = n_points / total_time if total_time > 0 else 0.0
        avg_latency_ms = (total_time / n_points * 1000.0) if n_points > 0 else 0.0

        stats = {
            "total_points": float(n_points),
            "elapsed_seconds": round(total_time, 4),
            "throughput_points_per_sec": round(points_per_sec, 2),
            "latency_ms_per_point": round(avg_latency_ms, 4),
        }
        return records, stats

    def reset(self) -> None:
        self.layer.reset()
