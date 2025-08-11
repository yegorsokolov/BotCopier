#!/usr/bin/env python3
"""Construct a symbol correlation graph.

This utility reads a CSV file containing price history for multiple symbols and
computes pairwise Pearson correlations. The resulting undirected graph is
written to JSON format suitable for consumption by :mod:`train_target_clone`.

The input CSV is expected to contain the columns ``timestamp``, ``symbol`` and
``price``. Timestamps are treated as categorical identifiers; they need only be
consistent across symbols.
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def build_graph(csv_path: Path, out_path: Path) -> dict:
    """Build correlation graph from price history.

    Parameters
    ----------
    csv_path: Path
        CSV file with columns ``timestamp``, ``symbol`` and ``price``.
    out_path: Path
        Destination path for graph JSON.
    """
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="timestamp", columns="symbol", values="price")
    corr = pivot.corr().fillna(0.0)
    symbols = list(corr.columns)
    edge_index = []
    edge_weight = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            w = float(corr.iloc[i, j])
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_weight.extend([w, w])
    graph = {
        "symbols": symbols,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)
    return graph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices", required=True, help="CSV file with timestamp,symbol,price")
    p.add_argument("--out", required=True, help="Output JSON file for graph")
    args = p.parse_args()
    build_graph(Path(args.prices), Path(args.out))


if __name__ == "__main__":
    main()
