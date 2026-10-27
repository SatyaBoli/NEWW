"""Data loader and streaming generator for NAB time-series benchmark."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from ..contracts import Point


@dataclass
class SeriesMetadata:
    series_id: str
    category: str
    series_name: str
    has_anomaly: bool
    num_rows: int
    start_time: str
    end_time: str


class NABDataLoader:
    """Robust streaming data loader for Numenta Anomaly Benchmark (NAB) datasets."""

    def __init__(self, data_path: Optional[str] = None, labels_path: Optional[str] = None) -> None:
        self.data_path = self._resolve_data_path(data_path)
        self.labels_path = self._resolve_labels_path(labels_path)
        self._df: Optional[pd.DataFrame] = None
        self._labels: Optional[dict[str, list[list[str]]]] = None

    @staticmethod
    def _resolve_data_path(custom_path: Optional[str]) -> str:
        candidates = [
            custom_path,
            os.environ.get("NAB_DATA_PATH"),
            os.path.join("data", "raw", "NAB_AI_Training_Dataset.csv"),
            os.path.join("data", "NAB_AI_Training_Dataset.csv"),
            os.path.join("data", "benchmark_test_data.csv"),
            "NAB_AI_Training_Dataset.csv",
            os.path.join(os.path.expanduser("~"), "Downloads", "NAB_AI_Training_Dataset.csv"),
            r"C:\Users\satya\Downloads\NAB_AI_Training_Dataset.csv",
        ]
        for p in candidates:
            if p and os.path.exists(p):
                return os.path.abspath(p)
        raise FileNotFoundError(
            f"NAB dataset not found. Checked: {[c for c in candidates if c]}. "
            "Please specify via `--data <path>` or set NAB_DATA_PATH environment variable."
        )

    @staticmethod
    def _resolve_labels_path(custom_path: Optional[str]) -> Optional[str]:
        candidates = [
            custom_path,
            os.environ.get("NAB_LABELS_PATH"),
            os.path.join("data", "combined_windows.json"),
            "combined_windows.json",
            os.path.join(os.path.expanduser("~"), "Downloads", "combined_windows.json"),
            r"C:\Users\satya\Downloads\combined_windows.json",
        ]
        for p in candidates:
            if p and os.path.exists(p):
                return os.path.abspath(p)
        return None

    def load_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self.data_path)
            
            # Robust schema normalization
            if "series_id" not in self._df.columns:
                self._df["series_id"] = "0"
            self._df["series_id"] = self._df["series_id"].astype(str)
            self._df["timestamp"] = self._df["timestamp"].astype(str)
            self._df["value"] = pd.to_numeric(self._df["value"], errors="coerce").fillna(0.0).astype(float)

            if "category" not in self._df.columns:
                self._df["category"] = "general"
            else:
                self._df["category"] = self._df["category"].astype(str)

            if "series_name" not in self._df.columns:
                self._df["series_name"] = self._df["series_id"].astype(str)
            else:
                self._df["series_name"] = self._df["series_name"].astype(str)

            if "has_anomaly_in_series" not in self._df.columns:
                self._df["has_anomaly_in_series"] = 0
            else:
                self._df["has_anomaly_in_series"] = pd.to_numeric(
                    self._df["has_anomaly_in_series"], errors="coerce"
                ).fillna(0).astype(int)

            # Sort strictly by timestamp and drop duplicate observations
            self._df = self._df.drop_duplicates(subset=["series_id", "timestamp"], keep="first")
            try:
                self._df["_dt"] = pd.to_datetime(self._df["timestamp"])
                self._df = self._df.sort_values(by=["series_id", "_dt"]).drop(columns=["_dt"]).reset_index(drop=True)
            except Exception:
                self._df = self._df.sort_values(by=["series_id", "timestamp"]).reset_index(drop=True)

        return self._df

    def load_labels(self) -> dict[str, list[list[str]]]:
        if self._labels is None:
            if self.labels_path and os.path.exists(self.labels_path):
                try:
                    with open(self.labels_path, "r", encoding="utf-8") as f:
                        self._labels = json.load(f)
                except Exception:
                    self._labels = {}
            else:
                self._labels = {}
        return self._labels

    def get_ground_truth_windows_by_series_id(self) -> dict[str, list[list[str]]]:
        """Maps series_id ('0', '1', ...) to ground truth anomaly windows."""
        df = self.load_df()
        raw_labels = self.load_labels()
        mapping = {}

        unique_series = df.drop_duplicates(subset=["series_id"])
        for _, row in unique_series.iterrows():
            sid = str(row["series_id"])
            cat = str(row.get("category", "general"))
            sname = str(row.get("series_name", sid))

            # NAB format key: "category/series_name.csv"
            key = f"{cat}/{sname}.csv"
            windows = raw_labels.get(key, [])
            if not windows:
                # Direct series_id fallback
                windows = raw_labels.get(sid, [])
            mapping[sid] = windows

        return mapping

    def list_series(self) -> list[SeriesMetadata]:
        df = self.load_df()
        result = []
        for sid, grp in df.groupby("series_id", sort=False):
            first = grp.iloc[0]
            last = grp.iloc[-1]
            meta = SeriesMetadata(
                series_id=str(sid),
                category=str(first.get("category", "general")),
                series_name=str(first.get("series_name", str(sid))),
                has_anomaly=bool(first.get("has_anomaly_in_series", 0) == 1),
                num_rows=len(grp),
                start_time=str(first["timestamp"]),
                end_time=str(last["timestamp"]),
            )
            result.append(meta)
        return result

    def stream_series(self, series_id: str) -> Iterator[Point]:
        df = self.load_df()
        subset = df[df["series_id"] == str(series_id)]
        for _, row in subset.iterrows():
            yield Point(
                series_id=str(row["series_id"]),
                timestamp=str(row["timestamp"]),
                value=float(row["value"]),
            )
