"""GPU training and fine-tuning engine for Causal TCN-AE on NVIDIA RTX 5050."""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data.loader import NABDataLoader
from .detectors.tcn_autoencoder import TCNAEModule


class TimeSeriesWindowDataset(Dataset):
    """Slices 1D time series into sliding windows for autoencoder reconstruction."""

    def __init__(self, series_list: list[np.ndarray], window_size: int = 32, stride: int = 4):
        self.windows = []
        for s in series_list:
            if len(s) < window_size:
                continue
            for i in range(0, len(s) - window_size + 1, stride):
                w = s[i : i + window_size]
                # Normalize window
                mu = np.mean(w)
                std = max(np.std(w), 1e-4)
                norm_w = (w - mu) / std
                self.windows.append(norm_w.astype(np.float32))

        self.data = np.stack(self.windows) if self.windows else np.empty((0, window_size), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns [1, window_size]
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)


def train_tcn_autoencoder(
    data_path: Optional[str] = None,
    output_checkpoint: str = "models/sentinel_tcn_best.pt",
    window_size: int = 32,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cuda",
) -> dict[str, float]:
    """Train Causal TCN-AE on NAB normal operating intervals using RTX 5050 GPU."""
    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)

    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        dev = torch.device("cpu")
        device_name = "CPU"

    print(f"[Training] Initializing on device: {dev} ({device_name})")

    # Ingest data
    loader = NABDataLoader(data_path=data_path)
    df = loader.load_df()

    # Select normal baseline series (artificialNoAnomaly + pre-anomaly segments)
    training_series = []
    normal_series_ids = df[df["category"] == "artificialNoAnomaly"]["series_id"].unique()
    for sid in normal_series_ids:
        vals = df[df["series_id"] == sid]["value"].values
        training_series.append(vals)

    # Add segments from other benchmark series
    other_sids = df[df["category"].isin(["realAWSCloudwatch", "realTraffic"])]["series_id"].unique()[:10]
    for sid in other_sids:
        vals = df[df["series_id"] == sid]["value"].values[:1500]  # First 1500 points are normal warmup
        training_series.append(vals)

    dataset = TimeSeriesWindowDataset(training_series, window_size=window_size, stride=4)
    print(f"[Training] Generated {len(dataset)} training windows from normal regimes.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TCNAEModule(input_dim=1, hidden_dim=32, latent_dim=16).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(dev)
            optimizer.zero_grad()

            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_checkpoint)

        gpu_mem = (
            torch.cuda.memory_allocated(0) / (1024 * 1024) if dev.type == "cuda" else 0.0
        )
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.6f} - Best: {best_loss:.6f} - GPU Mem: {gpu_mem:.1f}MB"
        )

    elapsed = time.time() - start_time
    print(
        f"[Training Complete] Best Loss: {best_loss:.6f} in {elapsed:.2f}s. Checkpoint saved: {output_checkpoint}"
    )

    return {
        "best_loss": float(best_loss),
        "epochs": epochs,
        "elapsed_seconds": float(elapsed),
        "device": str(dev),
    }


if __name__ == "__main__":
    train_tcn_autoencoder(epochs=10)
