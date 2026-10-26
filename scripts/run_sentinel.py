"""CLI entry point to execute Sentinel streaming pipeline on benchmark data."""
from __future__ import annotations

import argparse
import json
import os

from sentinel.data.loader import NABDataLoader
from sentinel.evaluation.event_f1 import evaluate_event_f1
from sentinel.output.submission_adapter import SubmissionAdapter
from sentinel.pipeline import SentinelPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Sentinel streaming anomaly detector")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML")
    parser.add_argument("--series-id", type=str, default=None, help="Process specific series_id (e.g., '11') or all")
    parser.add_argument("--output-csv", type=str, default="outputs/predictions.csv", help="Output submission CSV path")
    parser.add_argument("--evaluate", action="store_true", default=True, help="Compute Event-F1 metrics")

    args = parser.parse_args()

    print(f"[Sentinel CLI] Initializing pipeline from: {args.config}")
    pipeline = SentinelPipeline.from_yaml(args.config)
    loader = NABDataLoader()

    series_to_run = [args.series_id] if args.series_id is not None else [s.series_id for s in loader.list_series()]
    print(f"[Sentinel CLI] Processing {len(series_to_run)} series...")

    all_records = []
    total_time = 0.0
    total_points = 0

    for sid in series_to_run:
        stream = loader.stream_series(sid)
        records, stats = pipeline.process_stream(stream)
        all_records.extend(records)
        total_time += stats["elapsed_seconds"]
        total_points += int(stats["total_points"])

    print(f"[Sentinel CLI] Completed {total_points} observations in {total_time:.2f}s "
          f"({total_points / max(total_time, 1e-4):.1f} pts/sec, "
          f"{total_time / max(total_points, 1) * 1000:.3f} ms/pt).")

    # Write output predictions
    SubmissionAdapter.write_csv(all_records, args.output_csv)
    validation = SubmissionAdapter.validate_submission_file(args.output_csv)
    print(f"[Sentinel CLI] Submission Validation: {validation}")

    # Evaluate against ground truth
    if args.evaluate:
        gt_windows = loader.get_ground_truth_windows_by_series_id()
        metrics = evaluate_event_f1(all_records, gt_windows)
        print("\n=== Sentinel Evaluation Metrics ===")
        print(f"Event-Based F1 Score : {metrics.event_f1:.4f}")
        print(f"Event Precision       : {metrics.precision:.4f}")
        print(f"Event Recall          : {metrics.recall:.4f}")
        print(f"True Positive Events  : {metrics.true_positives}")
        print(f"False Positives       : {metrics.false_positives}")
        print(f"False Negatives       : {metrics.false_negatives}")
        print(f"Median Delay (seconds): {metrics.median_detection_delay_seconds:.1f}s")
        print("===================================\n")

        report_path = "outputs/evaluation_summary.json"
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
