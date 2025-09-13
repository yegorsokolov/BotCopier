#!/usr/bin/env python3
"""Pre-train a masked feature autoencoder.

This script trains a simple denoising autoencoder that randomly masks a
subset of input features and learns to reconstruct them.  The encoder
weights are saved to ``masked_encoder.pt`` so that downstream models can
use the compressed representation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train(
    data_file: Path,
    out_dir: Path,
    *,
    latent_dim: int = 8,
    mask_ratio: float = 0.3,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Path:
    """Train a masked autoencoder and return path to saved encoder.

    Parameters
    ----------
    data_file:
        CSV file containing feature columns. Any column starting with
        ``label`` is ignored.
    out_dir:
        Directory where ``masked_encoder.pt`` will be written.
    latent_dim:
        Dimension of the encoded feature space.
    mask_ratio:
        Fraction of features to randomly mask during training.
    epochs:
        Number of training epochs.
    batch_size:
        Training batch size.
    lr:
        Optimiser learning rate.
    """
    df = pd.read_csv(data_file)
    feature_cols = [c for c in df.columns if not c.startswith("label") and c != "profit"]
    X = df[feature_cols].to_numpy(dtype=float)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1]

    class AutoEncoder(nn.Module):
        def __init__(self, inp: int, latent: int) -> None:
            super().__init__()
            self.encoder = nn.Linear(inp, latent)
            self.decoder = nn.Linear(latent, inp)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
            return self.decoder(self.encoder(x))

    model = AutoEncoder(input_dim, latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for (batch,) in loader:
            mask = torch.rand_like(batch) < mask_ratio
            corrupted = batch.clone()
            corrupted[mask] = 0.0
            recon = model(corrupted)
            loss = ((recon - batch)[mask] ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    out_dir.mkdir(parents=True, exist_ok=True)
    enc_state = model.encoder.state_dict()
    save_path = out_dir / "masked_encoder.pt"
    torch.save(
        {
            "state_dict": enc_state,
            "architecture": [input_dim, latent_dim],
            "mask_ratio": mask_ratio,
        },
        save_path,
    )
    return save_path


def main() -> None:  # pragma: no cover - CLI wrapper
    p = argparse.ArgumentParser(description="Pretrain masked autoencoder")
    p.add_argument("data_file", help="CSV file with feature columns")
    p.add_argument("out_dir", help="Output directory for encoder")
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--mask-ratio", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    train(
        Path(args.data_file),
        Path(args.out_dir),
        latent_dim=args.latent_dim,
        mask_ratio=args.mask_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
