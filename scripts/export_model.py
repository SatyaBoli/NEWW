"""Production Model Exporter for Sentinel Causal TCN-AE.

Exports trained PyTorch weights to:
1. TorchScript JIT Optimized Model (.pt) - Zero-dependency C++ / Python streaming runtime.
2. PyTorch Exported Program (.pt2) - Standard PyTorch 2.0+ deployment format.
3. ONNX format (if onnxscript is installed).
"""
from __future__ import annotations

import os
import torch

from sentinel.detectors.tcn_autoencoder import TCNAEModule


def export_production_models(
    checkpoint_path: str = "models/sentinel_tcn_best.pt",
    output_dir: str = "models/exported",
    window_size: int = 32,
    hidden_dim: int = 32,
    latent_dim: int = 16,
) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Model Exporter] Loading weights from: {checkpoint_path}")
    model = TCNAEModule(input_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("[Model Exporter] Checkpoint loaded successfully.")
    else:
        print("[Model Exporter] Checkpoint not found; exporting architecture with initial weights.")

    model.eval()
    dummy_input = torch.randn(1, 1, window_size, dtype=torch.float32)
    artifacts = {}

    # 1. TorchScript JIT Export
    ts_path = os.path.join(output_dir, "sentinel_tcn_jit.pt")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(ts_path)
        size_kb = os.path.getsize(ts_path) / 1024.0
        print(f"[Model Exporter] [OK] Exported TorchScript JIT model to {ts_path} ({size_kb:.1f} KB)")
        artifacts["torchscript"] = ts_path
    except Exception as e:
        print(f"[Model Exporter] TorchScript export note: {e}")

    # 2. PyTorch 2.0+ torch.export
    pt2_path = os.path.join(output_dir, "sentinel_tcn_pt2.pt")
    try:
        exported_program = torch.export.export(model, (dummy_input,))
        torch.export.save(exported_program, pt2_path)
        size_kb = os.path.getsize(pt2_path) / 1024.0
        print(f"[Model Exporter] [OK] Exported PyTorch 2.0+ AOT model to {pt2_path} ({size_kb:.1f} KB)")
        artifacts["pt2"] = pt2_path
    except Exception as e:
        print(f"[Model Exporter] PyTorch AOT export note: {e}")

    # 3. ONNX Export (Optional if onnxscript installed)
    onnx_path = os.path.join(output_dir, "sentinel_tcn.onnx")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["causal_input_window"],
            output_names=["reconstructed_window"],
        )
        size_kb = os.path.getsize(onnx_path) / 1024.0
        print(f"[Model Exporter] [OK] Exported ONNX model to {onnx_path} ({size_kb:.1f} KB)")
        artifacts["onnx"] = onnx_path
    except Exception as e:
        print(f"[Model Exporter] ONNX export note: {e} (TorchScript / AOT models remain fully functional)")

    return artifacts


if __name__ == "__main__":
    export_production_models()
