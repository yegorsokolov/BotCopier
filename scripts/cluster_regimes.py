#!/usr/bin/env python3
"""Cluster historical feature vectors into market regimes.

This utility loads trade logs using the same feature extraction pipeline as
:mod:`train_target_clone` and runs a clustering algorithm (KMeans by default)
 to label distinct market regimes.  The resulting cluster centers and
scaling information are written to ``regime_model.json`` so that training and
real-time systems can detect the current regime.
"""
import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer

try:  # optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional
    hdbscan = None

# Reuse data loading and feature extraction from training script
from train_target_clone import _load_logs, _extract_features, _load_calendar  # type: ignore


def cluster_features(
    data_dir: Path,
    out_file: Path,
    clusters: int = 3,
    algorithm: str = "kmeans",
    corr_map: dict[str, list[str]] | None = None,
    calendar_events: list[tuple] | None = None,
    event_window: float = 60.0,
) -> None:
    """Cluster feature vectors and write model to ``out_file``."""
    rows_df, _, _ = _load_logs(data_dir)
    feats, *_ = _extract_features(
        rows_df.to_dict("records"),
        corr_map=corr_map,
        calendar_events=calendar_events,
        event_window=event_window,
    )
    if not feats:
        raise ValueError("No features found for clustering")

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(feats)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std

    if algorithm == "hdbscan" and hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, clusters))
        labels = clusterer.fit_predict(X_scaled)
        centers: List[np.ndarray] = []
        for lbl in sorted(set(labels)):
            if lbl == -1:
                continue  # noise
            centers.append(X_scaled[labels == lbl].mean(axis=0))
        centers_arr = np.vstack(centers) if centers else np.empty((0, X.shape[1]))
        algo_name = "hdbscan"
    else:
        km = KMeans(n_clusters=clusters, n_init=10, random_state=42)
        km.fit(X_scaled)
        centers_arr = km.cluster_centers_
        algo_name = "kmeans"

    model = {
        "feature_names": vec.get_feature_names_out().tolist(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "centers": centers_arr.astype(float).tolist(),
        "algorithm": algo_name,
    }
    out_file.write_text(json.dumps(model))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-file", type=Path, default=Path("regime_model.json"))
    p.add_argument("--clusters", type=int, default=3, help="number of regimes")
    p.add_argument(
        "--algorithm",
        choices=["kmeans", "hdbscan"],
        default="kmeans",
        help="clustering algorithm",
    )
    p.add_argument("--corr-symbols", help="comma separated correlated symbol pairs e.g. EURUSD:USDCHF")
    p.add_argument("--calendar-file", help="CSV file with columns time,impact[,id] for events")
    p.add_argument("--event-window", type=float, default=60.0, help="minutes around events to flag")
    args = p.parse_args()
    if args.corr_symbols:
        corr_map = {}
        for p_sym in args.corr_symbols.split(','):
            if ':' in p_sym:
                base, peer = p_sym.split(':', 1)
                corr_map.setdefault(base, []).append(peer)
    else:
        corr_map = None
    if args.calendar_file:
        events = _load_calendar(Path(args.calendar_file))
    else:
        events = None
    cluster_features(
        args.data_dir,
        args.out_file,
        args.clusters,
        args.algorithm,
        corr_map=corr_map,
        calendar_events=events,
        event_window=args.event_window,
    )


if __name__ == "__main__":
    main()
