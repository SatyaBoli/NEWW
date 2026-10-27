"""Export benchmark telemetry and model outputs to compact JSON for the standalone web monitor."""
from __future__ import annotations

import json
import os
import pandas as pd

from sentinel.config import AdaptationConfig, SentinelConfig
from sentinel.data.loader import NABDataLoader
from sentinel.detectors.ensemble import HybridEnsembleDetector
from sentinel.explanations.explainer import generate_explanation
from sentinel.pipeline import SentinelPipeline


def export_web_data():
    os.makedirs("docs/data", exist_ok=True)
    loader = NABDataLoader()
    gt_windows = loader.get_ground_truth_windows_by_series_id()
    all_series = loader.list_series()

    # 1. Corpus List (All 58 series)
    corpus_list = []
    for s in all_series:
        corpus_list.append({
            "series_id": s.series_id,
            "category": s.category,
            "series_name": s.series_name,
            "num_rows": s.num_rows,
            "has_anomaly": s.has_anomaly,
            "gt_windows": gt_windows.get(s.series_id, []),
            "start_time": s.start_time,
            "end_time": s.end_time,
            "is_benchmark": s.series_id in ["5", "7", "11", "13", "34"],
        })

    # 2. Benchmark Scorecard
    scorecard_df = pd.read_csv("outputs/experiment_log.csv")
    scorecard = scorecard_df.to_dict(orient="records")

    # 3. Stream Telemetry for 5 Benchmark Series
    benchmark_sids = ["11", "13", "34", "5", "7"]
    series_telemetry = {}
    flagged_events_by_sid = {}

    chk_path = "models/sentinel_specialist_best.pt"
    if not os.path.exists(chk_path):
        chk_path = "models/sentinel_tcn_best.pt"

    adapt_cfg = AdaptationConfig(
        enter_threshold=6.5,
        exit_threshold=2.5,
        threshold_mode="likelihood",
        min_event_length=3,
        cooldown_steps=10,
        warmup_steps=40,
        likelihood_long_window=600,
        drift_refresh_steps=40,
    )
    detector = HybridEnsembleDetector(checkpoint_path=chk_path, scale_floor=0.05, mad_window=60, ewma_alpha=0.12)
    pipeline = SentinelPipeline(SentinelConfig(adaptation=adapt_cfg))
    pipeline.detector = detector
    pipeline.layer.detector = detector

    for sid in benchmark_sids:
        print(f"[Export] Streaming Series {sid}...")
        pipeline.reset()
        stream_pts = list(loader.stream_series(sid))
        points = []
        events = {}
        sid_windows = gt_windows.get(sid, [])

        for idx, pt in enumerate(stream_pts):
            rec = pipeline.process_point(pt)
            points.append({
                "t": rec["timestamp"],
                "y": round(float(rec["evidence"]["observed"]), 4),
                "b": round(float(rec["evidence"]["baseline"]), 4),
                "s": round(float(rec["anomaly_score"]), 2),
                "th": round(float(rec["threshold"]), 2),
                "l": rec["label"],
                "e": rec.get("event_id") or "",
            })

            if rec["label"] == "ANOMALOUS":
                eid = rec.get("event_id")
                if eid and eid not in events:
                    expl = generate_explanation(rec)
                    ts = rec["timestamp"]
                    is_tp = any(w[0] <= ts <= w[1] for w in sid_windows if len(w) == 2)
                    events[eid] = {
                        "event_id": eid,
                        "timestamp": ts,
                        "observed": round(float(expl["observed_value"]), 4),
                        "baseline": round(float(expl["expected_baseline"]), 4),
                        "residual": round(float(expl["residual"]), 4),
                        "percent_deviation": round(float(expl["percent_deviation"]), 1),
                        "anomaly_score": round(float(rec["anomaly_score"]), 2),
                        "threshold": round(float(rec["threshold"]), 2),
                        "reason_codes": expl["reason_codes"],
                        "is_tp": is_tp,
                        "severity": "CRITICAL" if rec["anomaly_score"] >= 8.0 else ("HIGH" if rec["anomaly_score"] >= 6.5 else "MEDIUM"),
                    }

        series_telemetry[sid] = points
        flagged_events_by_sid[sid] = list(events.values())

    payload = {
        "version": "sentinel-0.2.0",
        "benchmark_sids": benchmark_sids,
        "corpus_list": corpus_list,
        "scorecard": scorecard,
        "series_telemetry": series_telemetry,
        "flagged_events": flagged_events_by_sid,
    }

    out_file = "docs/data/telemetry_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"[Export Complete] Saved {file_size_mb:.2f} MB payload to {out_file}")


if __name__ == "__main__":
    export_web_data()
