"""Train Model B (Infrastructure Domain Specialist) on NVIDIA RTX 5050 GPU.

Focuses training exclusively on server telemetry, machine sensors, and reference baselines:
- realAWSCloudwatch (17 series)
- realKnownCause (7 series)
- artificialNoAnomaly (5 series)
- artificialWithAnomaly (6 series)
Excludes social media tweets (realTweets) and online ad auctions (realAdExchange).
"""
from __future__ import annotations

import argparse
from sentinel.train import train_tcn_autoencoder


def main():
    parser = argparse.ArgumentParser(description="Train Model B: Infrastructure Specialist on RTX 5050 GPU")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Target computing device")
    parser.add_argument("--output", type=str, default="models/sentinel_specialist_best.pt", help="Checkpoint output path")
    args = parser.parse_args()

    specialist_categories = [
        "realAWSCloudwatch",
        "realKnownCause",
        "artificialNoAnomaly",
        "artificialWithAnomaly",
    ]

    print("[*] Launching Model B (Infrastructure Specialist) GPU Training...")
    print(f"[*] Target Categories: {specialist_categories}")

    metrics = train_tcn_autoencoder(
        output_checkpoint=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        categories=specialist_categories,
    )

    print("\n=======================================================")
    print("[*] Model B (Infrastructure Specialist) Training Complete!")
    print(f"[*] Best Validation Loss : {metrics['best_val_loss']:.6f}")
    print(f"[*] Test Anomaly Loss    : {metrics['test_anom_loss']:.6f}")
    print(f"[*] Separation Ratio     : {metrics['separation_ratio']:.2f}x")
    print(f"[*] Training Time        : {metrics['elapsed_seconds']:.2f}s on {metrics['device']}")
    print(f"[*] Checkpoint Saved To  : {args.output}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
