"""Unpack and format the full 58-series NAB benchmark corpus into Sentinel.

Creates:
1. data/raw/NAB_AI_Training_Dataset.csv (unified 6-column dataset across all 58 series, 365,558 rows)
2. data/nab/<category>/<series_name>.csv (original NAB directory tree matching numenta/NAB)
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


def unpack_nab():
    src_csv = Path(r"C:\Users\satya\Downloads\NAB_AI_Training_Dataset.csv")
    if not src_csv.exists():
        raise FileNotFoundError(f"Source file not found at: {src_csv}")

    print(f"Reading {src_csv}...")
    cols = ["series_id", "category", "series_name", "has_anomaly_in_series", "timestamp", "value"]
    df = pd.read_csv(src_csv, usecols=cols)

    os.makedirs("data/raw", exist_ok=True)
    out_csv = Path("data/raw/NAB_AI_Training_Dataset.csv")
    df.to_csv(out_csv, index=False)
    csv_size_mb = out_csv.stat().st_size / (1024 * 1024)
    print(f"[OK] Wrote {out_csv} ({csv_size_mb:.2f} MB, {len(df):,} rows, {df['series_id'].nunique()} series)")

    # Also unpack individual category directories to mirror numenta/NAB repository
    nab_dir = Path("data/nab")
    count = 0
    for (cat, sname), grp in df.groupby(["category", "series_name"], sort=False):
        target_dir = nab_dir / cat
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_file = target_dir / f"{sname}.csv"
        grp[["timestamp", "value"]].to_csv(csv_file, index=False)
        count += 1

    print(f"[OK] Successfully unpacked all {count} series into {nab_dir}/<category>/<series_name>.csv")


if __name__ == "__main__":
    unpack_nab()
