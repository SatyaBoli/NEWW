"""Controlled Architectural Component Ablation Study for Sentinel.

Isolates and quantifies the empirical contribution of each subsystem:
1. MAD Only: Pure non-parametric distribution model.
2. EWMA Only: Pure fast-adaptive trend tracker.
3. Causal TCN-AE Only: Deep neural reconstruction model without statistical priors.
4. Statistical Ensemble: Combination of MAD + EWMA without neural gating.
5. Full Sentinel Hybrid Ensemble: Calibrated multi-perspective fusion of Deep + Statistical.

Outputs results to outputs/ablation_log.csv.
"""
from __future__ import annotations

import os
import time
import pandas as pd

from sentinel.config import AdaptationConfig, DetectorConfig, SentinelConfig
from sentinel.data.loader import NABDataLoader
from sentinel.evaluation.event_f1 import evaluate_event_f1
from sentinel.pipeline import SentinelPipeline


def run_ablation_study(sample_series_ids: list[str] | None = None) -> pd.DataFrame:
    loader = NABDataLoader()
    gt_windows = loader.get_ground_truth_windows_by_series_id()

    if sample_series_ids is None:
        sample_series_ids = ["5", "7", "11", "13", "34"]

    print(f"[Ablation Study] Evaluating on {len(sample_series_ids)} series: {sample_series_ids}")
    eval_gt = {str(sid): gt_windows.get(str(sid), []) for sid in sample_series_ids}

    # Reference candidate adaptation
    cand_cfg = SentinelConfig.from_yaml("configs/candidate.yaml")

    ablations = [
        {
            "ablation_id": "A1_MAD_Only",
            "name": "Rolling Median + MAD Alone",
            "config": SentinelConfig(
                detector=DetectorConfig(detector_type="rolling_mad", window_size=60, scale_floor=0.05),
                adaptation=cand_cfg.adaptation,
            ),
            "description": "Baseline non-parametric point detector without trend or deep temporal awareness.",
        },
        {
            "ablation_id": "A2_EWMA_Only",
            "name": "Streaming EWMA Alone",
            "config": SentinelConfig(
                detector=DetectorConfig(detector_type="ewma", ewma_alpha=0.12, scale_floor=0.05),
                adaptation=cand_cfg.adaptation,
            ),
            "description": "Fast adaptive exponential filter without seasonal or reconstruction gating.",
        },
        {
            "ablation_id": "A3_TCN_AE_Only",
            "name": "Causal TCN-AE Alone (RTX 5050)",
            "config": SentinelConfig(
                detector=DetectorConfig(detector_type="autoencoder", ae_device="cuda", scale_floor=0.05),
                adaptation=cand_cfg.adaptation,
            ),
            "description": "Deep neural reconstruction model without statistical baseline anchoring.",
        },
        {
            "ablation_id": "A4_Statistical_Ensemble",
            "name": "Statistical Ensemble (MAD + EWMA, No Deep AE)",
            "config": SentinelConfig(
                detector=DetectorConfig(
                    detector_type="ensemble",
                    window_size=60,
                    ewma_alpha=0.12,
                    scale_floor=0.05,
                ),
                adaptation=cand_cfg.adaptation,
            ),
            "custom_weights": (0.60, 0.40, 0.0),
            "description": "Fusion of statistical baselines; neural TCN-AE weight set to 0.0.",
        },
        {
            "ablation_id": "A5_Full_Hybrid_Ensemble",
            "name": "Full Sentinel Hybrid Ensemble (Deep + Statistical + ADWIN)",
            "config": cand_cfg,
            "description": "Complete production architecture combining TCN-AE, MAD, EWMA, and guarded ADWIN.",
        },
    ]

    results = []

    for item in ablations:
        ab_id = item["ablation_id"]
        ab_name = item["name"]
        cfg = item["config"]
        print(f"\n--- Running Ablation: {ab_id} ({ab_name}) ---")

        pipeline = SentinelPipeline(cfg)
        if "custom_weights" in item and hasattr(pipeline.layer.detector, "weights"):
            pipeline.layer.detector.weights = item["custom_weights"]

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
            f"TP = {metrics.true_positives} | FP = {metrics.false_positives} | FN = {metrics.false_negatives} | "
            f"Delay = {metrics.median_detection_delay_seconds:.1f}s | "
            f"Throughput = {pts_per_sec:.0f} pts/sec"
        )

        results.append({
            "ablation_id": ab_id,
            "architecture": ab_name,
            "description": item["description"],
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
    out_csv = "outputs/ablation_log.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n[Ablation Study Complete] Saved log to: {out_csv}")
    print(results_df[["ablation_id", "event_f1", "precision", "recall", "true_positives", "false_positives"]].to_string())
    return results_df


if __name__ == "__main__":
    run_ablation_study()
