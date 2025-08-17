#!/usr/bin/env python3
"""Build a weighted symbol graph from precomputed correlation features.

This utility aggregates ``corr_*`` features exported by
``train_target_clone.py`` into an undirected weighted graph.  It then derives
simple node metrics (currently degree and PageRank centrality) which can be
consumed by the training pipeline as additional features.

The input file is expected to be a CSV or Parquet table containing at least the
columns ``symbol`` and one or more ``corr_<PEER>`` columns.  Each row
represents a trade for ``symbol`` with an observed correlation value to ``PEER``.

Example
-------
>>> build_graph(Path("features.csv"), Path("graph.json"))

The resulting JSON has the structure::

    {
        "symbols": ["EURUSD", "USDCHF"],
        "edge_index": [[0, 1], [1, 0]],
        "edge_weight": [0.9, 0.9],
        "metrics": {
            "degree": [0.9, 0.9],
            "pagerank": [0.5, 0.5]
        }
    }

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _compute_pagerank(adj: np.ndarray, alpha: float = 0.85, tol: float = 1e-6) -> np.ndarray:
    """Compute PageRank scores for a weighted adjacency matrix."""

    n = adj.shape[0]
    if n == 0:
        return np.zeros(0)
    pr = np.full(n, 1.0 / n)
    out_weight = adj.sum(axis=1)
    out_weight[out_weight == 0.0] = 1.0  # avoid division by zero
    for _ in range(100):
        prev = pr.copy()
        pr = (1 - alpha) / n + alpha * adj.T.dot(pr / out_weight)
        if np.abs(pr - prev).sum() < tol:
            break
    return pr


def build_graph(feat_path: Path, out_path: Path) -> dict:
    """Aggregate correlation features into a weighted graph and compute metrics."""

    if feat_path.suffix == ".parquet":
        df = pd.read_parquet(feat_path)
    else:
        df = pd.read_csv(feat_path)

    symbols = list(df["symbol"].dropna().unique())
    index = {s: i for i, s in enumerate(symbols)}
    weights: dict[tuple[str, str], list[float]] = {}

    for _, row in df.iterrows():
        base = row.get("symbol")
        if not isinstance(base, str):
            continue
        for col, val in row.items():
            if not col.startswith("corr_"):
                continue
            if pd.isna(val):
                continue
            peer = col[5:]
            if peer not in index:
                index[peer] = len(symbols)
                symbols.append(peer)
            pair = tuple(sorted((base, peer)))
            weights.setdefault(pair, []).append(float(val))

    n = len(symbols)
    adj = np.zeros((n, n), dtype=float)
    edge_index: list[list[int]] = []
    edge_weight: list[float] = []

    for (a, b), vals in weights.items():
        w = float(np.mean(np.abs(vals)))
        i, j = index[a], index[b]
        adj[i, j] = adj[j, i] = w
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_weight.extend([w, w])

    degree = adj.sum(axis=1)
    pagerank = _compute_pagerank(adj)

    graph = {
        "symbols": symbols,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "metrics": {
            "degree": degree.tolist(),
            "pagerank": pagerank.tolist(),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)
    return graph


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, help="CSV/Parquet file with corr_* features")
    p.add_argument("--out", required=True, help="Output JSON file for graph")
    args = p.parse_args()
    build_graph(Path(args.features), Path(args.out))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

