"""Champion vs. Challenger: Head-to-Head Comparative Evaluator.

Executes controlled A/B bake-off between:
- Model A (Champion): Global Generalist Causal TCN-AE (trained on all 58 series)
- Model B (Challenger): Infrastructure Specialist Causal TCN-AE (trained strictly on server/cloud/machine metrics)

Outputs a structured comparative scorecard to outputs/model_comparison_scorecard.csv.
"""
from __future__ import annotations

import os
import time
import pandas as pd

from sentinel.config import AdaptationConfig, DetectorConfig, SentinelConfig
from sentinel.data.loader import NABDataLoader
from sentinel.detectors.ensemble import HybridEnsembleDetector
from sentinel.detectors.tcn_autoencoder import CausalTCNAEDetector
from sentinel.evaluation.event_f1 import evaluate_event_f1
from sentinel.pipeline import SentinelPipeline


def run_model_comparison(test_series_ids: list[str] | None = None) -> pd.DataFrame:
    loader = NABDataLoader()
    gt_windows = loader.get_ground_truth_windows_by_series_id()

    if test_series_ids is None:
        # Representative infrastructure & server test streams
        test_series_ids = ["5", "7", "11", "13", "34"]

    print(f"\n=======================================================")
    print(f"[*] Sentinel Champion vs. Challenger Bake-Off")
    print(f"[*] Evaluating on {len(test_series_ids)} streams: {test_series_ids}")
    print(f"=======================================================\n")

    eval_gt = {str(sid): gt_windows.get(str(sid), []) for sid in test_series_ids}

    candidates = [
        {
            "model_id": "Model_A_TCN_Global",
            "model_name": "Model A (Global TCN-AE)",
            "checkpoint": "models/sentinel_tcn_best.pt",
            "detector_type": "autoencoder",
            "mode": "standalone_tcn",
        },
        {
            "model_id": "Model_B_TCN_Specialist",
            "model_name": "Model B (Infra Specialist TCN)",
            "checkpoint": "models/sentinel_specialist_best.pt",
            "detector_type": "autoencoder",
            "mode": "standalone_tcn",
        },
        {
            "model_id": "Model_A_Hybrid_Ensemble",
            "model_name": "Ensemble with Model A (Global)",
            "checkpoint": "models/sentinel_tcn_best.pt",
            "detector_type": "ensemble",
            "mode": "ensemble",
        },
        {
            "model_id": "Model_B_Hybrid_Ensemble",
            "model_name": "Ensemble with Model B (Specialist)",
            "checkpoint": "models/sentinel_specialist_best.pt",
            "detector_type": "ensemble",
            "mode": "ensemble",
        },
    ]

    results = []

    for c in candidates:
        print(f"--- Running: {c['model_name']} ---")
        adapt_cfg = AdaptationConfig(
            enter_threshold=6.5,
            exit_threshold=2.5,
            min_event_length=3,
            cooldown_steps=10,
            threshold_mode="likelihood",
            drift_enabled=True,
        )

        # Build custom detector pointing to specific checkpoint
        if c["mode"] == "standalone_tcn":
            detector = CausalTCNAEDetector(
                window_size=32,
                device="cuda",
                scale_floor=0.05,
                checkpoint_path=c["checkpoint"],
            )
        else:
            detector = HybridEnsembleDetector(
                checkpoint_path=c["checkpoint"],
                scale_floor=0.05,
            )

        pipeline = SentinelPipeline(SentinelConfig(adaptation=adapt_cfg))
        pipeline.detector = detector
        pipeline.layer.detector = detector

        all_records = []
        total_time = 0.0
        total_points = 0

        for sid in test_series_ids:
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
            f"  -> Event-F1 = {metrics.event_f1:.4f} | "
            f"Precision = {metrics.precision:.4f} | "
            f"Recall = {metrics.recall:.4f} | "
            f"FP = {metrics.false_positives} | "
            f"Speed = {pts_per_sec:.0f} pts/sec\n"
        )

        results.append({
            "model_id": c["model_id"],
            "model_name": c["model_name"],
            "checkpoint": c["checkpoint"],
            "event_f1": round(metrics.event_f1, 4),
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "true_positives": metrics.true_positives,
            "false_positives": metrics.false_positives,
            "false_negatives": metrics.false_negatives,
            "throughput_pts_sec": round(pts_per_sec, 1),
            "latency_ms_pt": round(latency_ms, 3),
            "total_points": total_points,
            "elapsed_seconds": round(total_time, 3),
        })

    df_scorecard = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/model_comparison_scorecard.csv"
    df_scorecard.to_csv(out_path, index=False)

    print("=======================================================================")
    print("[*] FINAL CHAMPION VS. CHALLENGER SCORECARD")
    print("=======================================================================")
    print(df_scorecard[["model_name", "event_f1", "precision", "recall", "false_positives", "throughput_pts_sec"]].to_string(index=False))
    print(f"\nSaved detailed comparison to: {out_path}\n")
    return df_scorecard


if __name__ == "__main__":
    run_model_comparison()
