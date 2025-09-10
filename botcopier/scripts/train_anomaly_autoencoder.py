#!/usr/bin/env python3
"""Train a tiny PCA-based autoencoder on trade features and store weights in model.json."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> np.ndarray:
    """Generate 6-dim feature vectors from raw trade data."""
    price = df["bid"].to_numpy()
    sl = price - 0.05
    tp = price + 0.05
    lots = np.ones_like(price)
    spread = (df["ask"] - df["bid"]).to_numpy()
    slippage = df["latency"].to_numpy() * 0.001
    X = np.vstack([price, sl, tp, lots, spread, slippage]).T
    return X.astype(np.float32)


def train_autoencoder(X: np.ndarray, latent_dim: int = 3):
    """Fit linear autoencoder via SVD and return weights/normalization."""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xs = (X - mean) / std
    _, _, vt = np.linalg.svd(Xs, full_matrices=False)
    W = vt[:latent_dim].T
    return mean, std, W


def main() -> None:
    p = argparse.ArgumentParser(description="Train simple autoencoder and update model.json")
    p.add_argument("input", type=Path, default=Path("tests/fixtures/trades_small.csv"), nargs="?")
    p.add_argument("--model", type=Path, default=Path("model.json"))
    p.add_argument("--latent", type=int, default=3)
    p.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="Number of rows per chunk when reading the input CSV (0 reads all rows)",
    )
    args = p.parse_args()

    if args.chunksize > 0:
        reader = pd.read_csv(args.input, sep=";", chunksize=args.chunksize)
        feats: list[np.ndarray] = []
        for chunk in reader:
            feats.append(build_features(chunk))
        X = np.vstack(feats) if feats else np.empty((0, 6), dtype=np.float32)
    else:
        df = pd.read_csv(args.input, sep=";")
        X = build_features(df)
    mean, std, W = train_autoencoder(X, latent_dim=args.latent)

    if args.model.exists():
        model = json.loads(args.model.read_text())
    else:
        model = {}
    model["ae_weights"] = W.flatten().tolist()
    model["ae_mean"] = mean.tolist()
    model["ae_std"] = std.tolist()
    model["ae_shape"] = [int(W.shape[0]), int(W.shape[1])]
    args.model.write_text(json.dumps(model, indent=2))


if __name__ == "__main__":
    main()
