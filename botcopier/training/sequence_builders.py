"""Sequence construction utilities for the training pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_symbol_context(
    symbols: Sequence[str],
    symbol_graph: Mapping[str, object] | str | Path | None,
) -> tuple[
    dict[str, int],
    list[str],
    np.ndarray,
    list[list[int]],
    dict[str, list[str]],
]:
    """Return embedding and neighbour metadata for ``symbols``."""

    unique_symbols = [str(sym) for sym in dict.fromkeys(symbols) if sym is not None]
    graph_data: Mapping[str, object] | None
    if symbol_graph is None:
        graph_data = None
    elif isinstance(symbol_graph, Mapping):
        graph_data = symbol_graph
    else:
        try:
            graph_data = json.loads(Path(symbol_graph).read_text())
        except Exception:
            logger.exception("Failed to load symbol graph from %s", symbol_graph)
            graph_data = None

    if graph_data:
        graph_symbols = list(graph_data.get("symbols", []))
        embeddings_map = graph_data.get("embeddings", {})
        if not graph_symbols and isinstance(embeddings_map, Mapping):
            graph_symbols = list(embeddings_map.keys())
        embedding_dim = int(graph_data.get("dimension", 0) or 0)
        if not embedding_dim and isinstance(embeddings_map, Mapping):
            first_vec = next(iter(embeddings_map.values()), None)
            if isinstance(first_vec, Sequence):
                embedding_dim = len(first_vec)
        if embedding_dim <= 0:
            embedding_dim = max(1, len(graph_symbols) or 1)
        symbol_names = list(graph_symbols)
        for sym in unique_symbols:
            if sym not in symbol_names:
                symbol_names.append(sym)
        emb_matrix = np.zeros((len(symbol_names), embedding_dim), dtype=float)
        if isinstance(embeddings_map, Mapping):
            for idx, sym in enumerate(symbol_names):
                vec = embeddings_map.get(sym)
                if isinstance(vec, Sequence):
                    arr = np.asarray(vec, dtype=float)
                    if arr.shape[0] != embedding_dim:
                        padded = np.zeros(embedding_dim, dtype=float)
                        length = min(arr.shape[0], embedding_dim)
                        padded[:length] = arr[:length]
                        emb_matrix[idx] = padded
                    else:
                        emb_matrix[idx] = arr
        edge_index = graph_data.get("edge_index")
        neighbor_lists: list[list[int]] = [[] for _ in symbol_names]
        if isinstance(edge_index, Sequence) and len(edge_index) == 2:
            src_iter, dst_iter = edge_index
            try:
                for src, dst in zip(src_iter, dst_iter):
                    src_i = int(src)
                    dst_i = int(dst)
                    if 0 <= src_i < len(graph_symbols) and 0 <= dst_i < len(graph_symbols):
                        neighbor_lists[src_i].append(dst_i)
            except TypeError:
                logger.debug("symbol graph edge_index not iterable; skipping")
        for idx in range(len(symbol_names)):
            if not neighbor_lists[idx]:
                neighbor_lists[idx] = [idx]
            else:
                seen: set[int] = set()
                ordered: list[int] = []
                if idx not in neighbor_lists[idx]:
                    neighbor_lists[idx].insert(0, idx)
                for item in neighbor_lists[idx]:
                    if 0 <= item < len(symbol_names) and item not in seen:
                        ordered.append(int(item))
                        seen.add(int(item))
                neighbor_lists[idx] = ordered or [idx]
    else:
        symbol_names = unique_symbols or []
        embedding_dim = max(1, len(symbol_names) or 1)
        emb_matrix = np.zeros((len(symbol_names), embedding_dim), dtype=float)
        for i in range(len(symbol_names)):
            emb_matrix[i, i % embedding_dim] = 1.0
        neighbor_lists = [[i] for i in range(len(symbol_names))]

    symbol_to_idx = {sym: i for i, sym in enumerate(symbol_names)}
    neighbor_order = {
        sym: [symbol_names[j] for j in neighbor_lists[idx]]
        for sym, idx in symbol_to_idx.items()
    }
    return symbol_to_idx, symbol_names, emb_matrix, neighbor_lists, neighbor_order


def build_window_sequences(
    X: np.ndarray,
    y: np.ndarray,
    profits: np.ndarray,
    sample_weight: np.ndarray,
    *,
    window_length: int,
    returns_df: pd.DataFrame | None = None,
    news_sequences: np.ndarray | None = None,
    symbols: np.ndarray | None = None,
    regime_features: np.ndarray | None = None,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    pd.DataFrame | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Construct rolling window sequences for sequence models."""

    if X.shape[0] < window_length:
        raise ValueError("Not enough samples for the requested window length")
    seq_list = [
        X[i - window_length + 1 : i + 1]
        for i in range(window_length - 1, X.shape[0])
    ]
    sequence_data = np.stack(seq_list, axis=0).astype(float)
    X = X[window_length - 1 :]
    y = y[window_length - 1 :]
    profits = profits[window_length - 1 :]
    sample_weight = sample_weight[window_length - 1 :]
    if regime_features is not None:
        regime_features = regime_features[window_length - 1 :]
    if returns_df is not None:
        returns_df = returns_df.iloc[window_length - 1 :].reset_index(drop=True)
    if news_sequences is not None:
        news_sequences = news_sequences[window_length - 1 :]
    if symbols is not None and symbols.size:
        symbols = symbols[window_length - 1 :]
    return (
        sequence_data,
        regime_features,
        X,
        y,
        profits,
        sample_weight,
        returns_df,
        news_sequences,
        symbols,
    )


__all__ = [
    "build_window_sequences",
    "prepare_symbol_context",
]
