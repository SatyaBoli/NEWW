"""High-Performance Edge Deployment: Export Causal TCN-AE to ONNX.

Exports PyTorch CausalTCNAEModule to an optimized ONNX computational graph
for micro-second latency inference via ONNX Runtime / TensorRT at the edge.
"""
from __future__ import annotations

import os
import torch

from sentinel.detectors.tcn_autoencoder import TCNAEModule


def export_to_onnx(
    checkpoint_path: str = "models/sentinel_tcn_best.pt",
    output_onnx_path: str = "models/sentinel_tcn.onnx",
    window_size: int = 32,
    hidden_dim: int = 32,
    latent_dim: int = 16,
) -> str:
    print(f"[ONNX Exporter] Loading trained PyTorch weights from: {checkpoint_path}")
    model = TCNAEModule(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("[ONNX Exporter] Checkpoint weights loaded successfully.")
    else:
        print("[ONNX Exporter] Warning: Checkpoint not found, exporting initialized architecture.")

    model.eval()

    # Create dummy streaming input: [batch_size=1, channels=1, seq_len=32]
    dummy_input = torch.randn(1, 1, window_size, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    print(f"[ONNX Exporter] Exporting computational graph to: {output_onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["causal_input_window"],
        output_names=["reconstructed_window"],
        dynamic_axes={
            "causal_input_window": {0: "batch_size"},
            "reconstructed_window": {0: "batch_size"},
        },
    )

    size_kb = os.path.getsize(output_onnx_path) / 1024.0
    print(f"[ONNX Exporter] Export complete! Model size: {size_kb:.1f} KB")
    return output_onnx_path


if __name__ == "__main__":
    export_to_onnx()
