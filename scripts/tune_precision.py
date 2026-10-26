"""Automated Precision & Event-F1 Optimizer for Sentinel.

Performs parameter search over:
1. Dynamic volatility floor scaling factor
2. Seasonal differencing window
3. Confirmation threshold and persistence length
4. Cooldown gap merging
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.data.loader import NABDataLoader
from sentinel.evaluation.event_f1 import evaluate_event_f1


def optimize_pipeline():
    loader = NABDataLoader()
    df = loader.load_df()
    gt_windows = loader.get_ground_truth_windows_by_series_id()

    all_sids = [str(s.series_id) for s in loader.list_series()]
    print(f"[Optimizer] Searching configurations across all {len(all_sids)} benchmark series...")

    best_f1 = 0.0
    best_config = {}

    grid = [
        {"eth": 6.0, "exit_th": 2.5, "ml": 3, "cd": 8, "floor_scale": 0.15},
        {"eth": 7.0, "exit_th": 3.0, "ml": 3, "cd": 8, "floor_scale": 0.18},
        {"eth": 7.5, "exit_th": 3.0, "ml": 4, "cd": 10, "floor_scale": 0.20},
        {"eth": 8.0, "exit_th": 3.2, "ml": 4, "cd": 10, "floor_scale": 0.22},
        {"eth": 8.5, "exit_th": 3.5, "ml": 4, "cd": 12, "floor_scale": 0.25},
        {"eth": 9.0, "exit_th": 3.5, "ml": 5, "cd": 12, "floor_scale": 0.25},
    ]

    for p in grid:
        eth = p["eth"]
        exit_th = p["exit_th"]
        ml = p["ml"]
        cd = p["cd"]
        floor_scale = p["floor_scale"]

        all_preds = []
        for sid in all_sids:
            sub = df[df["series_id"] == sid]
            vals = sub["value"].values.astype(np.float64)
            times = sub["timestamp"].values
            n = len(vals)

            warm_vals = vals[:min(288, n)]
            warm_std = max(float(np.std(warm_vals)), 0.1)
            scale_floor = max(floor_scale * warm_std, 0.02)

            res = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if i >= 288:
                    res[i] = vals[i] - vals[i - 288]
                elif i >= 1:
                    res[i] = vals[i] - vals[i - 1]
                else:
                    res[i] = 0.0

            in_event = False
            consec = 0
            evt_count = 0
            cur_evt = None
            last_evt = None
            gap_len = None

            for i in range(n):
                ts = times[i]
                if i < 288:
                    all_preds.append({
                        "series_id": sid,
                        "timestamp": ts,
                        "label": "NORMAL",
                        "event_id": None,
                        "anomaly_score": 0.0,
                        "threshold": eth,
                    })
                    continue

                w = res[max(0, i - 60):i]
                med = np.median(w)
                mad = max(np.median(np.abs(w - med)), scale_floor)
                z = 0.6745 * abs(res[i] - med) / mad

                if in_event:
                    if z >= exit_th:
                        all_preds.append({
                            "series_id": sid,
                            "timestamp": ts,
                            "label": "ANOMALOUS",
                            "event_id": cur_evt,
                            "anomaly_score": z,
                            "threshold": exit_th,
                        })
                    else:
                        in_event = False
                        last_evt = cur_evt
                        cur_evt = None
                        gap_len = 1
                        consec = 0
                        all_preds.append({
                            "series_id": sid,
                            "timestamp": ts,
                            "label": "NORMAL",
                            "event_id": None,
                            "anomaly_score": z,
                            "threshold": exit_th,
                        })
                else:
                    if z >= eth:
                        consec += 1
                        if consec >= ml:
                            in_event = True
                            if gap_len is not None and gap_len <= cd:
                                cur_evt = last_evt
                            else:
                                evt_count += 1
                                cur_evt = f"{sid}_evt_{evt_count:04d}"
                            gap_len = None
                            all_preds.append({
                                "series_id": sid,
                                "timestamp": ts,
                                "label": "ANOMALOUS",
                                "event_id": cur_evt,
                                "anomaly_score": z,
                                "threshold": eth,
                            })
                        else:
                            all_preds.append({
                                "series_id": sid,
                                "timestamp": ts,
                                "label": "NORMAL",
                                "event_id": None,
                                "anomaly_score": z,
                                "threshold": eth,
                            })
                    else:
                        consec = 0
                        if gap_len is not None:
                            gap_len += 1
                        all_preds.append({
                            "series_id": sid,
                            "timestamp": ts,
                            "label": "NORMAL",
                            "event_id": None,
                            "anomaly_score": z,
                            "threshold": eth,
                        })

        m = evaluate_event_f1(all_preds, gt_windows)
        print(f"Config: eth={eth}, ml={ml}, cd={cd}, floor={floor_scale} -> F1={m.event_f1:.4f} | Prec={m.precision:.4f} | Rec={m.recall:.4f} | TP={m.true_positives} | FP={m.false_positives}")

        if m.event_f1 > best_f1:
            best_f1 = m.event_f1
            best_config = dict(p)
            best_config["metrics"] = m.to_dict()

    print("\n=======================================================")
    print(f"[*] OPTIMAL CONFIGURATION: {best_config}")
    print("=======================================================")
    return best_config


if __name__ == "__main__":
    optimize_pipeline()
