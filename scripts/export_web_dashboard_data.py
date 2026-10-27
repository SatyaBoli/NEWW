"""Export Sentinel pipeline results as a JSON blob for the web dashboard.

Reads the project's output CSV files and generates a consolidated JSON
file that can be used to update the embedded data in the HTML dashboard.

Usage:
    python scripts/export_web_dashboard_data.py
"""
import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_csv(path):
    """Read a CSV file and return list of dicts."""
    full = os.path.join(BASE, path)
    if not os.path.exists(full):
        print(f"  ⚠ {path} not found, skipping.")
        return []
    with open(full, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    print("\n🛡️  Sentinel Dashboard Data Exporter")
    print("=" * 42)

    data = {
        "project": "Sentinel — Adaptive Time-Series Anomaly Detection",
        "hackathon": "Utkrishta 2026 — Challenge 2",
        "models": {},
        "experiments": [],
        "ablations": [],
    }

    # 1. Model comparison scorecard
    print("\n📊 Reading model comparison scorecard...")
    rows = read_csv("outputs/model_comparison_scorecard.csv")
    for r in rows:
        mid = r.get("model_id", r.get("model_name", "unknown"))
        data["models"][mid] = {
            "name": r.get("model_name", mid),
            "checkpoint": r.get("checkpoint", ""),
            "event_f1": float(r.get("event_f1", 0)),
            "precision": float(r.get("precision", 0)),
            "recall": float(r.get("recall", 0)),
            "true_positives": int(r.get("true_positives", 0)),
            "false_positives": int(r.get("false_positives", 0)),
            "throughput_pts_sec": float(r.get("throughput_pts_sec", 0)),
            "latency_ms_pt": float(r.get("latency_ms_pt", 0)),
        }
    print(f"  ✓ {len(rows)} model configurations loaded.")

    # 2. Experiment log
    print("\n🔬 Reading experiment log...")
    rows = read_csv("outputs/experiment_log.csv")
    for r in rows:
        data["experiments"].append({
            "id": r.get("experiment_id", ""),
            "name": r.get("model_name", ""),
            "detector": r.get("detector_type", ""),
            "event_f1": float(r.get("event_f1", 0)),
            "precision": float(r.get("precision", 0)),
            "recall": float(r.get("recall", 0)),
            "false_positives": int(r.get("false_positives", 0)),
            "throughput": float(r.get("throughput_pts_sec", 0)),
            "latency_ms": float(r.get("latency_ms_pt", 0)),
        })
    print(f"  ✓ {len(rows)} experiments loaded.")

    # 3. Ablation log
    print("\n🧪 Reading ablation log...")
    rows = read_csv("outputs/ablation_log.csv")
    for r in rows:
        data["ablations"].append({
            "id": r.get("ablation_id", r.get("experiment_id", "")),
            "name": r.get("model_name", ""),
            "event_f1": float(r.get("event_f1", 0)),
            "precision": float(r.get("precision", 0)),
            "recall": float(r.get("recall", 0)),
            "false_positives": int(r.get("false_positives", 0)),
        })
    print(f"  ✓ {len(rows)} ablation runs loaded.")

    # 4. Evaluation summary
    eval_path = os.path.join(BASE, "outputs/evaluation_summary.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            data["evaluation"] = json.load(f)
        print("\n📈 Evaluation summary loaded.")

    # Write output
    out_path = os.path.join(BASE, "outputs", "dashboard_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Dashboard data written to: {out_path}")
    print(f"   Size: {os.path.getsize(out_path):,} bytes")


if __name__ == "__main__":
    main()
