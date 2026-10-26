"""Repeatable experiment registry & baseline bake-off benchmarking suite.

Executes controlled comparisons between B0 (MAD), B1 (EWMA), B2 (iForest),
B3 (TCN-AE on GPU), and B4 (Hybrid Ensemble) on identical test streams.
"""
from __future__ import annotations

import os
import time
import pandas as pd

from sentinel.config import AdaptationConfig, DetectorConfig, SentinelConfig
from sentinel.data.loader import NABDataLoader
from sentinel.evaluation.event_f1 import evaluate_event_f1
from sentinel.pipeline import SentinelPipeline


def run_benchmark_experiments(sample_series_ids: list[str] | None = None) -> pd.DataFrame:
    loader = NABDataLoader()
    gt_windows = loader.get_ground_truth_windows_by_series_id()

    if sample_series_ids is None:
        # Benchmark subset representing various anomaly types:
        # art_daily_jumpsup (7), ec2_cpu (11), ec2_disk (19), ambient_temp (34), nyc_taxi (38), twitter (48)
        sample_series_ids = ["5", "7", "11", "13", "19", "28", "34", "38", "41", "48"]

    print(f"[Benchmark] Evaluating on {len(sample_series_ids)} representative series: {sample_series_ids}")

    experiments = [
        {
            "exp_id": "E1_B0_MAD",
            "name": "Rolling Median + MAD",
            "detector_type": "rolling_mad",
            "threshold_mode": "robust_z",
            "enter_threshold": 3.5,
            "exit_threshold": 2.0,
            "drift_enabled": False,
        },
        {
            "exp_id": "E2_B1_EWMA",
            "name": "Streaming EWMA (alpha=0.15)",
            "detector_type": "ewma",
            "threshold_mode": "robust_z",
            "enter_threshold": 3.5,
            "exit_threshold": 2.0,
            "drift_enabled": False,
        },
        {
            "exp_id": "E3_B2_iForest",
            "name": "Causal Isolation Forest",
            "detector_type": "iforest",
            "threshold_mode": "robust_z",
            "enter_threshold": 3.0,
            "exit_threshold": 1.8,
            "drift_enabled": False,
        },
        {
            "exp_id": "E4_B3_TCN_AE",
            "name": "Causal TCN-AE (CUDA)",
            "detector_type": "autoencoder",
            "threshold_mode": "likelihood",
            "enter_threshold": 3.2,
            "exit_threshold": 1.8,
            "drift_enabled": True,
        },
        {
            "exp_id": "E5_B4_HybridEnsemble",
            "name": "Hybrid Ensemble + ADWIN Guard",
            "detector_type": "ensemble",
            "threshold_mode": "likelihood",
            "enter_threshold": 3.2,
            "exit_threshold": 1.8,
            "drift_enabled": True,
        },
    ]

    results = []

    for exp in experiments:
        print(f"\n--- Running Experiment: {exp['exp_id']} ({exp['name']}) ---")
        det_cfg = DetectorConfig(detector_type=exp["detector_type"], ae_device="cuda")
        adapt_cfg = AdaptationConfig(
            enter_threshold=exp["enter_threshold"],
            exit_threshold=exp["exit_threshold"],
            threshold_mode=exp["threshold_mode"],
            drift_enabled=exp["drift_enabled"],
        )
        pipeline = SentinelPipeline(SentinelConfig(detector=det_cfg, adaptation=adapt_cfg))

        all_records = []
        total_time = 0.0
        total_points = 0

        for sid in sample_series_ids:
            stream = loader.stream_series(sid)
            recs, stats = pipeline.process_stream(stream)
            all_records.extend(recs)
            total_time += stats["elapsed_seconds"]
            total_points += int(stats["total_points"])

        metrics = evaluate_event_f1(all_records, gt_windows)
        pts_per_sec = total_points / max(total_time, 1e-4)
        latency_ms = (total_time / max(total_points, 1)) * 1000.0

        print(
            f"Result: Event-F1 = {metrics.event_f1:.4f} | "
            f"Precision = {metrics.precision:.4f} | "
            f"Recall = {metrics.recall:.4f} | "
            f"Delay = {metrics.median_detection_delay_seconds:.1f}s | "
            f"Throughput = {pts_per_sec:.0f} pts/sec"
        )

        results.append({
            "experiment_id": exp["exp_id"],
            "model_name": exp["name"],
            "detector_type": exp["detector_type"],
            "threshold_mode": exp["threshold_mode"],
            "event_f1": round(metrics.event_f1, 4),
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "median_delay_sec": round(metrics.median_detection_delay_seconds, 1),
            "throughput_pts_sec": round(pts_per_sec, 1),
            "latency_ms_pt": round(latency_ms, 3),
            "total_points": total_points,
            "elapsed_seconds": round(total_time, 3),
        })

    results_df = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    out_csv = "outputs/experiment_log.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[Benchmark Complete] Saved experiment scorecard to: {out_csv}")
    print(results_df.to_string())
    return results_df


if __name__ == "__main__":
    run_benchmark_experiments()
