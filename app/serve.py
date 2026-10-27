"""Enterprise HTTP & API Server for the Sentinel Anomaly Detection System.

Provides:
  - Static file serving for the interactive Web Dashboard (app/index.html)
  - REST API /api/models: List of all 10 detector models and their capabilities
  - REST API /api/analysis: Full bake-off scorecard, ablation study, and benchmark metrics
  - REST API /api/run: Real-time execution of any selected detector on any NAB series
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.parse
import webbrowser
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BANNER = r"""
 ╔══════════════════════════════════════════════════════════════════════╗
 ║  [SENTINEL] Adaptive Anomaly Detection Engine & API Server           ║
 ║                                                                      ║
 ║  • Web Dashboard:   http://localhost:{port:<5}/index.html               ║
 ║  • REST Endpoints:  /api/models, /api/analysis, /api/run             ║
 ║  • Live Models:     Model A, Model B, Hybrid, MAD, EWMA, iForest...  ║
 ╚══════════════════════════════════════════════════════════════════════╝
"""

# Lazy-loaded data loader and pipeline components
_LOADER = None


def get_data_loader():
    global _LOADER
    if _LOADER is None:
        try:
            from sentinel.data.loader import NABDataLoader
            csv_path = os.path.join(PROJECT_ROOT, "data", "raw", "NAB_AI_Training_Dataset.csv")
            if os.path.exists(csv_path):
                _LOADER = NABDataLoader(csv_path)
        except Exception as e:
            print(f"  [Warning] Failed to initialize NABDataLoader: {e}")
    return _LOADER


MODEL_CATALOG = [
    {
        "id": "model_b_hybrid",
        "name": "Hybrid Ensemble (Model B Specialist - Champion)",
        "badge": "Champion",
        "type": "Multi-Model Consensus",
        "description": "Weighted fusion of Rolling MAD (0.45), EWMA (0.25), and Infrastructure Specialist TCN-AE (0.30) with guarded anti-contamination updates.",
        "event_f1": 0.3750,
        "precision": 0.3750,
        "recall": 0.3750,
        "false_positives": 5,
        "latency_ms": 1.11,
        "checkpoint": "models/sentinel_specialist_best.pt",
    },
    {
        "id": "model_a_hybrid",
        "name": "Hybrid Ensemble (Model A Global Generalist)",
        "badge": "Challenger",
        "type": "Multi-Model Consensus",
        "description": "Ensemble incorporating Global Generalist TCN-AE trained across all NAB domains.",
        "event_f1": 0.3077,
        "precision": 0.4000,
        "recall": 0.2500,
        "false_positives": 3,
        "latency_ms": 1.16,
        "checkpoint": "models/sentinel_tcn_best.pt",
    },
    {
        "id": "model_b_tcn",
        "name": "Model B: Infrastructure Specialist TCN-AE (CUDA)",
        "badge": "Neural",
        "type": "Deep Neural Network",
        "description": "Temporal Convolutional Autoencoder with causal dilated 1D convolutions trained on AWS CloudWatch and server telemetry.",
        "event_f1": 0.1290,
        "precision": 0.0870,
        "recall": 0.2500,
        "false_positives": 21,
        "latency_ms": 0.82,
        "checkpoint": "models/sentinel_specialist_best.pt",
    },
    {
        "id": "model_a_tcn",
        "name": "Model A: Global Generalist TCN-AE (CUDA)",
        "badge": "Neural",
        "type": "Deep Neural Network",
        "description": "TCN Autoencoder trained across multi-domain streams including social media and ad exchanges.",
        "event_f1": 0.0870,
        "precision": 0.0667,
        "recall": 0.1250,
        "false_positives": 14,
        "latency_ms": 1.16,
        "checkpoint": "models/sentinel_tcn_best.pt",
    },
    {
        "id": "stat_ensemble",
        "name": "Statistical Ensemble (MAD + EWMA)",
        "badge": "Statistical",
        "type": "Statistical Consensus",
        "description": "Combination of Median Absolute Deviation and Streaming EWMA without deep neural autoencoder.",
        "event_f1": 0.2609,
        "precision": 0.2000,
        "recall": 0.3750,
        "false_positives": 12,
        "latency_ms": 0.35,
        "checkpoint": None,
    },
    {
        "id": "rolling_mad",
        "name": "Rolling Median + MAD Baseline (B0)",
        "badge": "Baseline",
        "type": "Statistical Baseline",
        "description": "Non-parametric robust rolling z-score using Median Absolute Deviation. Fast, outlier-resistant.",
        "event_f1": 0.1333,
        "precision": 0.0811,
        "recall": 0.3750,
        "false_positives": 34,
        "latency_ms": 0.315,
        "checkpoint": None,
    },
    {
        "id": "ewma",
        "name": "Streaming EWMA Baseline (B1)",
        "badge": "Baseline",
        "type": "Statistical Baseline",
        "description": "Causal Exponentially Weighted Moving Average tracking dynamic mean & variance in O(1) time.",
        "event_f1": 0.0833,
        "precision": 0.0469,
        "recall": 0.3750,
        "false_positives": 61,
        "latency_ms": 0.033,
        "checkpoint": None,
    },
    {
        "id": "iforest",
        "name": "Causal Isolation Forest Baseline (B2)",
        "badge": "Baseline",
        "type": "Ensemble Trees",
        "description": "Sliding window tree partitioning isolating anomalies in causal lag feature space.",
        "event_f1": 0.1818,
        "precision": 0.3333,
        "recall": 0.1250,
        "false_positives": 2,
        "latency_ms": 3.687,
        "checkpoint": None,
    },
    {
        "id": "lof",
        "name": "Causal Local Outlier Factor Baseline (B3)",
        "badge": "Baseline",
        "type": "Density Outlier",
        "description": "Evaluates local density deviations compared to k-nearest neighbors in causal history.",
        "event_f1": 0.1359,
        "precision": 0.0737,
        "recall": 0.8750,
        "false_positives": 88,
        "latency_ms": 1.960,
        "checkpoint": None,
    },
    {
        "id": "svm",
        "name": "Causal One-Class SVM Baseline (B4)",
        "badge": "Baseline",
        "type": "Kernel Boundary",
        "description": "Non-linear RBF kernel hyper-plane separating nominal operating manifolds from outliers.",
        "event_f1": 0.1600,
        "precision": 0.0952,
        "recall": 0.5000,
        "false_positives": 38,
        "latency_ms": 1.299,
        "checkpoint": None,
    },
]


def instantiate_pipeline(model_id: str):
    """Instantiate a SentinelPipeline configured for the given model_id."""
    from sentinel.config import AdaptationConfig, DetectorConfig, SentinelConfig
    from sentinel.detectors.ensemble import HybridEnsembleDetector
    from sentinel.detectors.tcn_autoencoder import CausalTCNAEDetector
    from sentinel.pipeline import SentinelPipeline

    adapt_cfg = AdaptationConfig(
        enter_threshold=6.5,
        exit_threshold=2.5,
        threshold_mode="likelihood",
        min_event_length=3,
        cooldown_steps=10,
        warmup_steps=30,
    )

    chk_b = os.path.join(PROJECT_ROOT, "models", "sentinel_specialist_best.pt")
    chk_a = os.path.join(PROJECT_ROOT, "models", "sentinel_tcn_best.pt")

    if model_id == "model_b_hybrid":
        chk = chk_b if os.path.exists(chk_b) else chk_a
        det = HybridEnsembleDetector(checkpoint_path=chk, scale_floor=0.05, mad_window=60, ewma_alpha=0.12)
        pipe = SentinelPipeline(SentinelConfig(adaptation=adapt_cfg))
        pipe.detector = det
        pipe.layer.detector = det
        return pipe
    elif model_id == "model_a_hybrid":
        chk = chk_a if os.path.exists(chk_a) else chk_b
        det = HybridEnsembleDetector(checkpoint_path=chk, scale_floor=0.05, mad_window=60, ewma_alpha=0.12)
        pipe = SentinelPipeline(SentinelConfig(adaptation=adapt_cfg))
        pipe.detector = det
        pipe.layer.detector = det
        return pipe
    elif model_id == "model_b_tcn":
        chk = chk_b if os.path.exists(chk_b) else chk_a
        det = CausalTCNAEDetector(window_size=32, device="cuda", scale_floor=0.05, checkpoint_path=chk)
        pipe = SentinelPipeline(SentinelConfig(adaptation=adapt_cfg))
        pipe.detector = det
        pipe.layer.detector = det
        return pipe
    elif model_id == "model_a_tcn":
        chk = chk_a if os.path.exists(chk_a) else chk_b
        det = CausalTCNAEDetector(window_size=32, device="cuda", scale_floor=0.05, checkpoint_path=chk)
        pipe = SentinelPipeline(SentinelConfig(adaptation=adapt_cfg))
        pipe.detector = det
        pipe.layer.detector = det
        return pipe
    elif model_id == "stat_ensemble":
        det = HybridEnsembleDetector(checkpoint_path=None, weights=(0.60, 0.40, 0.0), scale_floor=0.05, mad_window=60, ewma_alpha=0.12)
        pipe = SentinelPipeline(SentinelConfig(adaptation=adapt_cfg))
        pipe.detector = det
        pipe.layer.detector = det
        return pipe
    elif model_id == "rolling_mad":
        cfg = SentinelConfig(detector=DetectorConfig(detector_type="rolling_mad", window_size=60, scale_floor=0.05), adaptation=adapt_cfg)
        return SentinelPipeline(cfg)
    elif model_id == "ewma":
        cfg = SentinelConfig(detector=DetectorConfig(detector_type="ewma", ewma_alpha=0.12, scale_floor=0.05), adaptation=adapt_cfg)
        return SentinelPipeline(cfg)
    elif model_id == "iforest":
        cfg = SentinelConfig(detector=DetectorConfig(detector_type="iforest", iforest_n_estimators=30, scale_floor=0.05), adaptation=adapt_cfg)
        return SentinelPipeline(cfg)
    elif model_id == "lof":
        cfg = SentinelConfig(detector=DetectorConfig(detector_type="lof", lof_n_neighbors=20, scale_floor=0.05), adaptation=adapt_cfg)
        return SentinelPipeline(cfg)
    elif model_id == "svm":
        cfg = SentinelConfig(detector=DetectorConfig(detector_type="svm", svm_nu=0.05, scale_floor=0.05), adaptation=adapt_cfg)
        return SentinelPipeline(cfg)
    else:
        return SentinelPipeline.from_yaml(os.path.join(PROJECT_ROOT, "configs", "candidate.yaml"))


class SentinelHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler with static file serving and JSON REST API endpoints."""

    def __init__(self, *args, **kwargs):
        # Serve static files from app/ directory
        app_dir = os.path.join(PROJECT_ROOT, "app")
        super().__init__(*args, directory=app_dir, **kwargs)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            url = urllib.parse.urlparse(self.path)
            path = url.path.rstrip("/")

            if path == "/api/models":
                self._send_json({"models": MODEL_CATALOG, "count": len(MODEL_CATALOG)})
                return

            elif path == "/api/analysis":
                json_file = os.path.join(PROJECT_ROOT, "outputs", "dashboard_data.json")
                if os.path.exists(json_file):
                    with open(json_file, "r", encoding="utf-8") as f:
                        self._send_json(json.load(f))
                else:
                    self._send_json({"error": "dashboard_data.json not found", "models": MODEL_CATALOG})
                return

            elif path == "/api/series":
                loader = get_data_loader()
                if loader:
                    gt_wins = loader.get_ground_truth_windows_by_series_id()
                    s_list = [
                        {
                            "series_id": s.series_id,
                            "series_name": s.series_name,
                            "category": s.category,
                            "num_points": getattr(s, "num_rows", 0),
                            "has_anomaly": getattr(s, "has_anomaly", False),
                            "num_windows": len(gt_wins.get(str(s.series_id), [])),
                            "key": f"{s.category}/{s.series_name}.csv",
                        }
                        for s in loader.list_series()
                    ]
                    self._send_json({"series": s_list, "count": len(s_list)})
                else:
                    self._send_json({"series": [], "count": 0})
                return

            # Default static file handler
            return super().do_GET()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path.rstrip("/")

        if path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body_raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(body_raw)
            except Exception:
                payload = {}

            series_id = str(payload.get("series_id", "11"))
            model_id = payload.get("model_id", "model_b_hybrid")
            max_points = int(payload.get("max_points", 0))

            loader = get_data_loader()
            if not loader:
                self._send_json({"error": "NAB dataset loader not initialized"}, status=500)
                return

            try:
                # Clean series identifier
                clean_target = series_id.replace(".csv", "").split("/")[-1].strip()
                matched_sid = None

                # Search by exact id, key, or clean series_name
                for s in loader.list_series():
                    if s.series_id == series_id:
                        matched_sid = s.series_id
                        break
                    full_key = f"{s.category}/{s.series_name}.csv"
                    if full_key == series_id:
                        matched_sid = s.series_id
                        break
                    if s.series_name == clean_target:
                        matched_sid = s.series_id
                        break
                    if clean_target and clean_target in s.series_name:
                        matched_sid = s.series_id
                        break

                if not matched_sid:
                    # Default fallback
                    matched_sid = loader.list_series()[0].series_id

                series_id = matched_sid
                all_pts = list(loader.stream_series(series_id))

                if max_points > 0:
                    pts = all_pts[:max_points]
                else:
                    pts = all_pts

                # Build pipeline and run stream
                t_start = time.perf_counter()
                pipeline = instantiate_pipeline(model_id)
                records, stats = pipeline.process_stream(pts)
                elapsed_sec = time.perf_counter() - t_start

                # Ground truth windows for checking true positives
                gt_windows = loader.get_ground_truth_windows_by_series_id().get(str(series_id), [])

                t_list = []
                x_list = []
                base_list = []
                score_list = []
                thr_list = []
                label_list = []
                alerts_map: dict[str, dict[str, Any]] = {}

                for idx, r in enumerate(records):
                    t_list.append(r["timestamp"])
                    obs_val = round(float(r["evidence"]["observed"]), 2)
                    base_val = round(float(r["evidence"]["baseline"]), 2)
                    score_val = round(float(r["anomaly_score"]), 2)
                    thr_val = round(float(r["threshold"]), 2)
                    is_anom = r["label"] == "ANOMALOUS"

                    x_list.append(obs_val)
                    base_list.append(base_val)
                    score_list.append(score_val)
                    thr_list.append(thr_val)
                    label_list.append(1 if is_anom else 0)

                    eid = r.get("event_id")
                    if is_anom and eid:
                        if eid not in alerts_map:
                            ts_str = r["timestamp"]
                            # Check if timestamp falls in any gt window
                            is_tp = False
                            for gw in gt_windows:
                                if len(gw) == 2 and gw[0] <= ts_str <= gw[1]:
                                    is_tp = True
                                    break
                            alerts_map[eid] = {
                                "start": ts_str,
                                "end": ts_str,
                                "peak": ts_str,
                                "label": "ANOMALOUS",
                                "true_positive": is_tp,
                                "peak_score": score_val,
                                "evidence": f"Model {model_id} flagged value {obs_val} vs baseline {base_val} (score {score_val} vs threshold {thr_val})",
                                "ix": [idx, idx, idx],
                            }
                        else:
                            alerts_map[eid]["end"] = r["timestamp"]
                            alerts_map[eid]["ix"][1] = idx
                            if score_val > alerts_map[eid]["peak_score"]:
                                alerts_map[eid]["peak_score"] = score_val
                                alerts_map[eid]["peak"] = r["timestamp"]
                                alerts_map[eid]["ix"][2] = idx

                alerts_list = list(alerts_map.values())
                tp_count = sum(1 for a in alerts_list if a["true_positive"])
                fp_count = len(alerts_list) - tp_count

                # Resolve display model info
                m_info = next((m for m in MODEL_CATALOG if m["id"] == model_id), None)

                response_data = {
                    "series_id": str(series_id),
                    "model_id": model_id,
                    "model_name": m_info["name"] if m_info else model_id,
                    "total_points": len(pts),
                    "elapsed_seconds": round(elapsed_sec, 4),
                    "latency_ms_per_point": round((elapsed_sec / max(len(pts), 1)) * 1000.0, 3),
                    "throughput_pts_sec": round(len(pts) / max(elapsed_sec, 1e-4), 1),
                    "alerts_count": len(alerts_list),
                    "true_positives": tp_count,
                    "false_positives": fp_count,
                    "t": t_list,
                    "x": x_list,
                    "baseline": base_list,
                    "s": score_list,
                    "thr": thr_list,
                    "flag": label_list,
                    "alerts": alerts_list,
                    "win": gt_windows,
                }
                self._send_json(response_data)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": "Endpoint not found"}, status=404)


def main():
    parser = argparse.ArgumentParser(description="Sentinel Web Dashboard & API Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    app_dir = os.path.join(PROJECT_ROOT, "app")
    os.chdir(app_dir)

    # Warm up data loader in background thread
    threading.Thread(target=get_data_loader, daemon=True).start()

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("", args.port), SentinelHandler) as httpd:
        url = f"http://localhost:{args.port}/index.html"
        print(BANNER.format(port=args.port))
        if not args.no_browser:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✓ Sentinel Server gracefully stopped.")


if __name__ == "__main__":
    main()
