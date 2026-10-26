"""CLI script to train and fine-tune Causal TCN-AE on local RTX 5050 GPU."""
from __future__ import annotations

import argparse
from sentinel.train import train_tcn_autoencoder


def main():
    parser = argparse.ArgumentParser(description="Train Sentinel Causal TCN-AE on GPU")
    parser.add_argument("--data-path", type=str, default=None, help="Path to NAB CSV dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="models/sentinel_tcn_best.pt", help="Checkpoint output")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda/cpu)")

    args = parser.parse_args()

    results = train_tcn_autoencoder(
        data_path=args.data_path,
        output_checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    print("Training Results:", results)


if __name__ == "__main__":
    main()
