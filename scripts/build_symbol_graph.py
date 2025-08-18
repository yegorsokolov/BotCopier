#!/usr/bin/env python3
"""Build a weighted symbol graph from precomputed correlation features.

This utility aggregates ``corr_*`` features exported by
``train_target_clone.py`` into an undirected weighted graph.  It derives
simple node metrics (currently degree and PageRank centrality) and, when
``torch_geometric`` is available, Node2Vec embeddings.  The resulting graph can
be written to JSON or Parquet and consumed by the training pipeline as
additional features.

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

try:  # pragma: no cover - optional dependency
    import torch
    from torch_geometric.nn import Node2Vec  # type: ignore
    _HAS_PYG = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    Node2Vec = None  # type: ignore
    _HAS_PYG = False

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

    graph: dict = {
        "symbols": symbols,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "metrics": {
            "degree": degree.tolist(),
            "pagerank": pagerank.tolist(),
        },
    }

    # Optional Node2Vec embeddings
    if _HAS_PYG and edge_index:
        try:  # pragma: no cover - heavy optional dependency
            ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            node2vec = Node2Vec(
                ei, embedding_dim=8, walk_length=5, context_size=3, walks_per_node=10
            )
            optim = torch.optim.Adam(node2vec.parameters(), lr=0.01)
            loader = node2vec.loader(batch_size=32, shuffle=True)
            for _ in range(10):
                for pos_rw, neg_rw in loader:
                    optim.zero_grad()
                    loss = node2vec.loss(pos_rw, neg_rw)
                    loss.backward()
                    optim.step()
            emb = node2vec.embedding.weight.detach().cpu().numpy()
            graph["embedding_dim"] = emb.shape[1]
            graph["embeddings"] = {
                sym: emb[i].astype(float).tolist() for i, sym in enumerate(symbols)
            }
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df_out = pd.DataFrame({
            "symbol": symbols,
            "degree": degree,
            "pagerank": pagerank,
        })
        if graph.get("embeddings"):
            emb = np.array([graph["embeddings"][s] for s in symbols], dtype=float)
            for j in range(emb.shape[1]):
                df_out[f"emb_{j}"] = emb[:, j]
        try:  # pragma: no cover - optional dependency
            df_out.to_parquet(out_path)
        except Exception:  # fall back to JSON if parquet support missing
            df_out.to_json(out_path, orient="records")
    else:
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)
    return graph


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, help="CSV/Parquet file with corr_* features")
    p.add_argument("--out", required=True, help="Output JSON or Parquet file for graph")
    args = p.parse_args()
    build_graph(Path(args.features), Path(args.out))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

