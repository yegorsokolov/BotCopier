#!/usr/bin/env python3
"""Build a weighted symbol graph from rolling correlations of trading symbols.

The script ingests a CSV or Parquet table containing either pre-computed
``corr_*`` feature columns or raw ``symbol``/``price`` data.  When only price
data is supplied rolling correlations are computed on the fly.  These
correlations are aggregated into an undirected weighted graph from which simple
node metrics (currently degree and PageRank centrality) are derived.  When
``torch_geometric`` is available, optional Node2Vec embeddings are also
estimated.  The resulting graph can be written to JSON or Parquet and consumed
by the training pipeline as additional features.

If pre-computed ``corr_*`` columns are present the script simply aggregates
them.  Otherwise it expects ``symbol`` and ``price`` columns and will compute
rolling correlations across all observed symbols.

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
from itertools import combinations

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


def build_graph(feat_path: Path, out_path: Path, corr_window: int = 20) -> dict:
    """Aggregate correlation features into a weighted graph and compute metrics.

    Parameters
    ----------
    feat_path : Path
        Input CSV/Parquet file.  When ``corr_*`` columns are present their
        values are aggregated directly.  Otherwise the file is expected to
        contain at least ``symbol`` and ``price`` columns from which rolling
        correlations are computed.
    out_path : Path
        Destination for the generated graph (JSON or Parquet).
    corr_window : int, optional
        Rolling window size used when computing correlations from price data.
    """

    if feat_path.suffix == ".parquet":
        df = pd.read_parquet(feat_path)
    else:
        df = pd.read_csv(feat_path)

    weights: dict[tuple[str, str], list[float]] = {}
    coint_beta: dict[tuple[str, str], float] = {}

    corr_cols = [c for c in df.columns if c.startswith("corr_")]
    has_price_cols = "symbol" in df.columns and "price" in df.columns

    pivot = None
    if has_price_cols:
        idx_col = "event_time" if "event_time" in df.columns else None
        if idx_col is None:
            df = df.copy()
            df["_idx"] = np.arange(len(df))
            idx_col = "_idx"
        pivot = df.pivot_table(index=idx_col, columns="symbol", values="price")
        pivot.sort_index(inplace=True)

    if corr_cols:
        symbols = list(df["symbol"].dropna().unique())
        index = {s: i for i, s in enumerate(symbols)}

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
    elif pivot is not None:
        symbols = [s for s in pivot.columns if str(s) != "nan"]
        index = {s: i for i, s in enumerate(symbols)}
        for a, b in combinations(symbols, 2):
            series = pivot[a].rolling(corr_window).corr(pivot[b]).dropna()
            vals = np.abs(series.values)
            if len(vals) == 0:
                continue
            pair = tuple(sorted((a, b)))
            weights[pair] = vals.tolist()
    else:
        raise ValueError(
            "input must contain corr_* columns or symbol/price columns"
        )

    # Estimate cointegration betas when price data available
    if pivot is not None:
        symbols = [s for s in pivot.columns if str(s) != "nan"]
        for a, b in combinations(symbols, 2):
            series = pivot[[a, b]].dropna()
            if len(series) < 2:
                continue
            x = series[b].values
            y = series[a].values
            X = np.vstack([x, np.ones(len(x))]).T
            beta, _ = np.linalg.lstsq(X, y, rcond=None)[0]
            coint_beta[(a, b)] = float(beta)
            Xr = np.vstack([y, np.ones(len(y))]).T
            beta_rev, _ = np.linalg.lstsq(Xr, x, rcond=None)[0]
            coint_beta[(b, a)] = float(beta_rev)

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
    if coint_beta:
        coint_map: dict[str, dict[str, float]] = {}
        for (a, b), beta in coint_beta.items():
            coint_map.setdefault(a, {})[b] = beta
        graph["cointegration"] = coint_map

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
    p.add_argument(
        "--features",
        required=True,
        help="CSV/Parquet file with corr_* features or symbol/price columns",
    )
    p.add_argument(
        "--out", required=True, help="Output JSON or Parquet file for graph"
    )
    p.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window for correlation when corr_* columns are absent",
    )
    args = p.parse_args()
    build_graph(Path(args.features), Path(args.out), corr_window=args.window)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

