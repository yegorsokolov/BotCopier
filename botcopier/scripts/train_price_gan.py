#!/usr/bin/env python3
"""Train a simple GAN on historical tick data.

The script reads a CSV file containing bid/ask quotes and learns a tiny
Generative Adversarial Network (GAN) that models short sequences of
price changes.  The resulting generator network can later be used to
produce synthetic sequences which are blended into the training data of
``train_target_clone.py``.

The implementation intentionally keeps the network architecture and
training loop compact so that the script remains lightweight and easy to
understand.  It is not meant to be a state of the art price generator but
serves as a deterministic, reproducible baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn, optim


class Generator(nn.Module):
    """Simple fully connected generator."""

    def __init__(self, latent_dim: int, seq_len: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple pass
        return self.net(z)


class Discriminator(nn.Module):
    """Small discriminator distinguishing real from fake sequences."""

    def __init__(self, seq_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple pass
        return self.net(x)


def load_tick_sequences(path: Path, seq_len: int) -> np.ndarray:
    """Return sliding windows of mid price returns."""

    import pandas as pd

    df = pd.read_csv(path)
    if {"bid", "ask"}.issubset(df.columns):
        mid = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
    else:
        # Fallback: treat any single price column as mid
        price_cols = [c for c in df.columns if c not in {"time"}]
        if not price_cols:
            raise ValueError("tick file must contain bid/ask or price column")
        mid = df[price_cols[0]].astype(float)
    returns = mid.diff().fillna(0.0).to_numpy(dtype=np.float32)
    seqs = []
    for i in range(len(returns) - seq_len):
        seqs.append(returns[i : i + seq_len])
    if not seqs:
        raise ValueError("not enough data for requested sequence length")
    return np.stack(seqs)


def train_gan(seqs: np.ndarray, epochs: int, latent_dim: int) -> Generator:
    """Train GAN on provided sequences."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = seqs.shape[1]
    gen = Generator(latent_dim, seq_len).to(device)
    disc = Discriminator(seq_len).to(device)

    opt_g = optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = optim.Adam(disc.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    data = torch.tensor(seqs, device=device)
    batch_size = min(64, len(data))

    for epoch in range(epochs):  # pragma: no cover - training loop is best effort
        perm = torch.randperm(len(data))
        for i in range(0, len(data), batch_size):
            real = data[perm[i : i + batch_size]]
            bsz = real.size(0)

            # Train discriminator
            z = torch.randn(bsz, latent_dim, device=device)
            fake = gen(z).detach()
            loss_d = criterion(disc(real), torch.ones(bsz, 1, device=device)) + criterion(
                disc(fake), torch.zeros(bsz, 1, device=device)
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # Train generator
            z = torch.randn(bsz, latent_dim, device=device)
            fake = gen(z)
            loss_g = criterion(disc(fake), torch.ones(bsz, 1, device=device))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

    gen.eval()
    return gen


def save_model(gen: Generator, path: Path) -> None:
    """Persist generator weights and meta data."""

    checkpoint = {
        "state_dict": gen.state_dict(),
        "latent_dim": gen.latent_dim,
        "seq_len": gen.seq_len,
    }
    torch.save(checkpoint, path)


def load_model(path: Path) -> Generator:
    """Load a previously trained :class:`Generator`.

    The helper is intentionally lightweight so it can also be reused in
    unit tests where a tiny GAN is trained for a handful of steps.  It
    returns the generator in evaluation mode.
    """

    checkpoint = torch.load(path, map_location="cpu")
    gen = Generator(checkpoint["latent_dim"], checkpoint["seq_len"])
    gen.load_state_dict(checkpoint["state_dict"])
    gen.eval()
    return gen


def sample_sequences(path: Path, n: int, seed: int | None = None) -> np.ndarray:
    """Generate ``n`` synthetic return sequences from a stored model."""

    if seed is not None:
        torch.manual_seed(seed)
    gen = load_model(path)
    with torch.no_grad():
        z = torch.randn(n, gen.latent_dim)
        seqs = gen(z).cpu().numpy()
    return seqs


def main() -> None:  # pragma: no cover - CLI entry point
    p = argparse.ArgumentParser(description="Train a GAN on tick data")
    p.add_argument("tick_file", help="CSV file containing bid/ask quotes")
    p.add_argument("--out", default="price_gan.pt", help="Output model file")
    p.add_argument("--seq-len", type=int, default=20, help="Sequence length")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs")
    p.add_argument("--latent-dim", type=int, default=16, help="Latent dimension")
    args = p.parse_args()

    seqs = load_tick_sequences(Path(args.tick_file), args.seq_len)
    gen = train_gan(seqs, args.epochs, args.latent_dim)
    save_model(gen, Path(args.out))


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
