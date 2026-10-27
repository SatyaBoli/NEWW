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
        # 5 diverse representative series:
        # art_daily_flatmiddle (5), art_daily_jumpsup (7), ec2_cpu (11), ec2_cpu_5f5533 (13), ambient_temp (34)
        sample_series_ids = ["5", "7", "11", "13", "34"]

    print(f"[Benchmark] Evaluating on {len(sample_series_ids)} representative series: {sample_series_ids}")
    eval_gt = {str(sid): gt_windows.get(str(sid), []) for sid in sample_series_ids}

    experiments = [
        {
            "exp_id": "E1_B0_MAD",
            "name": "Rolling Median + MAD",
            "detector_type": "rolling_mad",
            "threshold_mode": "robust_z",
            "enter_threshold": 6.0,
            "exit_threshold": 2.5,
            "drift_enabled": False,
            "min_event_length": 3,
            "cooldown_steps": 10,
        },
        {
            "exp_id": "E2_B1_EWMA",
            "name": "Streaming EWMA (alpha=0.15)",
            "detector_type": "ewma",
            "threshold_mode": "robust_z",
            "enter_threshold": 6.0,
            "exit_threshold": 2.5,
            "drift_enabled": False,
            "min_event_length": 3,
            "cooldown_steps": 10,
        },
        {
            "exp_id": "E3_B2_iForest",
            "name": "Causal Isolation Forest",
            "detector_type": "iforest",
            "threshold_mode": "robust_z",
            "enter_threshold": 3.5,
            "exit_threshold": 1.8,
            "drift_enabled": False,
            "min_event_length": 2,
            "cooldown_steps": 8,
        },
        {
            "exp_id": "E3b_LOF",
            "name": "Causal Local Outlier Factor",
            "detector_type": "lof",
            "threshold_mode": "robust_z",
            "enter_threshold": 3.8,
            "exit_threshold": 1.8,
            "drift_enabled": False,
            "min_event_length": 2,
            "cooldown_steps": 8,
        },
        {
            "exp_id": "E3c_OCSVM",
            "name": "Causal One-Class SVM (RBF)",
            "detector_type": "svm",
            "threshold_mode": "robust_z",
            "enter_threshold": 3.5,
            "exit_threshold": 1.8,
            "drift_enabled": False,
            "min_event_length": 2,
            "cooldown_steps": 8,
        },
        {
            "exp_id": "E4_B3_TCN_AE",
            "name": "Causal TCN-AE (CUDA)",
            "detector_type": "autoencoder",
            "threshold_mode": "likelihood",
            "enter_threshold": 5.5,
            "exit_threshold": 2.2,
            "drift_enabled": True,
            "min_event_length": 3,
            "cooldown_steps": 10,
        },
        {
            "exp_id": "E5_B4_HybridEnsemble",
            "name": "Hybrid Ensemble + ADWIN Guard",
            "detector_type": "ensemble",
            "threshold_mode": "likelihood",
            "enter_threshold": 6.5,
            "exit_threshold": 2.5,
            "drift_enabled": True,
            "min_event_length": 3,
            "cooldown_steps": 10,
            "window_size": 60,
            "ewma_alpha": 0.12,
            "warmup_steps": 40,
            "regime_shift_patience": 40,
            "likelihood_long_window": 600,
            "drift_refresh_steps": 40,
        },
    ]

    results = []

    for exp in experiments:
        print(f"\n--- Running Experiment: {exp['exp_id']} ({exp['name']}) ---")
        det_cfg = DetectorConfig(
            detector_type=exp["detector_type"],
            ae_device="cuda",
            scale_floor=0.05,
            window_size=exp.get("window_size", 50),
            ewma_alpha=exp.get("ewma_alpha", 0.15),
            iforest_n_estimators=exp.get("iforest_n_estimators", 30),
            lof_n_neighbors=exp.get("lof_n_neighbors", 20),
            svm_nu=exp.get("svm_nu", 0.05),
        )
        adapt_cfg = AdaptationConfig(
            enter_threshold=exp["enter_threshold"],
            exit_threshold=exp["exit_threshold"],
            threshold_mode=exp["threshold_mode"],
            drift_enabled=exp["drift_enabled"],
            min_event_length=exp.get("min_event_length", 3),
            cooldown_steps=exp.get("cooldown_steps", 10),
            warmup_steps=exp.get("warmup_steps", 30),
            regime_shift_patience=exp.get("regime_shift_patience", 50),
            likelihood_long_window=exp.get("likelihood_long_window", 500),
            drift_refresh_steps=exp.get("drift_refresh_steps", 50),
            detector_version="sentinel-deep-0.2.0" if exp["detector_type"] == "ensemble" else "sentinel-baseline-0.1.0",
        )
        pipeline = SentinelPipeline(SentinelConfig(detector=det_cfg, adaptation=adapt_cfg))

        all_records = []
        total_time = 0.0
        total_points = 0

        for sid in sample_series_ids:
            pipeline.reset()
            stream = loader.stream_series(sid)
            recs, stats = pipeline.process_stream(stream)
            all_records.extend(recs)
            total_time += stats["elapsed_seconds"]
            total_points += int(stats["total_points"])

        metrics = evaluate_event_f1(all_records, eval_gt)
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
