#!/usr/bin/env python3
"""Pre-train simple contrastive encoder on tick sequences.

This script implements a small SimCLR-style objective over windows of
price differences.  The resulting encoder is a single linear layer so the
weights can easily be embedded into generated MQL4 code.  For convenience
the trained weights are stored in ``encoder.pt`` and an equivalent ONNX
file ``encoder.onnx`` is exported if ``torch.onnx`` is available.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class TickDataset(Dataset):
    """Dataset producing windows of price differences."""

    def __init__(self, tick_dir: Path, window: int):
        self.seqs: List[torch.Tensor] = []
        for file in tick_dir.glob("ticks_*.csv"):
            prices: List[float] = []
            with open(file, newline="") as f:
                reader = csv.DictReader(f, delimiter=";")
                for r in reader:
                    try:
                        prices.append(float(r.get("bid", 0) or 0))
                    except Exception:
                        continue
            if len(prices) <= window:
                continue
            diffs = torch.tensor(prices).diff().float()
            for i in range(len(diffs) - window):
                self.seqs.append(diffs[i : i + window])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:  # pragma: no cover - trivial
        return self.seqs[idx]


class Encoder(nn.Module):
    """Very small encoder with optional projection head."""

    def __init__(self, window: int, dim: int):
        super().__init__()
        self.enc = nn.Linear(window, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        h = self.enc(x)
        z = nn.functional.normalize(self.proj(torch.relu(h)), dim=1)
        return h, z


def augment(x: torch.Tensor) -> torch.Tensor:
    noise = 0.01 * torch.randn_like(x)
    return x + noise


def simclr_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    batch_size = z1.size(0)
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels, labels])
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)
    return nn.functional.cross_entropy(sim, labels)


def train(tick_dir: Path, out_dir: Path, window: int, dim: int, epochs: int, batch: int) -> None:
    ds = TickDataset(tick_dir, window)
    if len(ds) == 0:
        raise ValueError(f"no tick sequences found in {tick_dir}")
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(window, dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for seq in dl:
            seq = seq.to(device)
            v1 = augment(seq)
            v2 = augment(seq)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = simclr_loss(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"state_dict": model.enc.state_dict(), "window": window, "dim": dim}
    torch.save(state, out_dir / "encoder.pt")
    try:  # optional ONNX export
        dummy = torch.randn(1, window)
        torch.onnx.export(model.enc, dummy, out_dir / "encoder.onnx", input_names=["x"], output_names=["z"])
    except Exception:
        pass


def main() -> None:  # pragma: no cover - CLI wrapper
    p = argparse.ArgumentParser(description="Contrastive pretraining from ticks")
    p.add_argument("tick_dir", type=Path, help="directory with ticks_*.csv files")
    p.add_argument("out_dir", type=Path, help="where to write encoder.pt")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()
    train(args.tick_dir, args.out_dir, args.window, args.dim, args.epochs, args.batch)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
