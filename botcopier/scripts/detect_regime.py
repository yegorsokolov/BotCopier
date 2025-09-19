#!/usr/bin/env python3
"""Cluster historical feature vectors into market regimes and label each sample.

This utility reuses the feature extraction from :mod:`train_target_clone` to
load trade logs, cluster the resulting feature vectors using either KMeans or
HDBSCAN and write the regime assignments along with the cluster centers to a
JSON file.
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

try:  # optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional
    hdbscan = None

from botcopier.data.loading import _load_calendar, _load_logs
from botcopier.features.engineering import (
    FeatureConfig,
    _extract_features,
    configure_cache,
)


def detect_regimes(
    data_dir: Path,
    out_file: Path,
    clusters: int = 3,
    algorithm: str = "kmeans",
    model_json: Optional[Path] = None,
    assignments_csv: Optional[Path] = None,
    corr_map: dict[str, list[str]] | None = None,
    calendar_events: list[tuple] | None = None,
    event_window: float = 60.0,
) -> None:
    """Cluster feature vectors and write regime IDs and centers to ``out_file``.

    Parameters
    ----------
    data_dir:
        Directory containing trade logs.
    out_file:
        File where raw clustering output will be written.
    clusters:
        Number of clusters to detect when using KMeans.
    algorithm:
        Clustering algorithm name (``kmeans`` or ``hdbscan``).
    model_json:
        Optional path to a model JSON file. When provided, the detected
        regime information and gating weights are merged into this file so
        that downstream utilities can embed them into generated strategies.
    """
    feature_config = configure_cache(FeatureConfig())
    rows_df, _, _ = _load_logs(data_dir, feature_config=feature_config)
    feats, *_ = _extract_features(
        rows_df.to_dict("records"),
        corr_map=corr_map,
        calendar_events=calendar_events,
        event_window=event_window,
        config=feature_config,
    )
    if not feats:
        raise ValueError("No features found for clustering")

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(feats)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std

    labels: np.ndarray
    centers_arr: np.ndarray
    if algorithm == "hdbscan" and hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, clusters))
        labels = clusterer.fit_predict(X_scaled)
        centers: List[np.ndarray] = []
        for lbl in sorted(set(labels)):
            if lbl == -1:
                continue  # noise
            centers.append(X_scaled[labels == lbl].mean(axis=0))
        centers_arr = np.vstack(centers) if centers else np.empty((0, X.shape[1]))
    else:
        km = KMeans(n_clusters=clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        centers_arr = km.cluster_centers_

    gating_clf = LogisticRegression(max_iter=200, multi_class="multinomial")
    gating_clf.fit(X_scaled, labels)

    model = {
        "feature_names": vec.get_feature_names_out().tolist(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "centers": centers_arr.astype(float).tolist(),
        "labels": labels.astype(int).tolist(),
        "algorithm": algorithm,
        "gating_coefficients": gating_clf.coef_.astype(float).tolist(),
        "gating_intercepts": gating_clf.intercept_.astype(float).tolist(),
    }
    out_file.write_text(json.dumps(model))

    if assignments_csv is not None:
        rows_df.assign(regime=labels).to_csv(assignments_csv, index=False)

    if model_json is not None:
        update = {
            "regime_feature_names": model["feature_names"],
            "regime_centers": model["centers"],
            "gating_coefficients": model["gating_coefficients"],
            "gating_intercepts": model["gating_intercepts"],
            "mean": model["mean"],
            "std": model["std"],
        }
        existing = {}
        if model_json.exists():
            try:
                existing = json.loads(model_json.read_text())
            except Exception:
                existing = {}
        existing.update(update)
        model_json.write_text(json.dumps(existing))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-file", type=Path, default=Path("regime_detection.json"))
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument(
        "--algorithm",
        choices=["kmeans", "hdbscan"],
        default="kmeans",
        help="clustering algorithm",
    )
    p.add_argument("--model-json", type=Path, default=Path("model.json"))
    p.add_argument(
        "--assignments", type=Path, help="CSV file to write per-sample regime IDs"
    )
    p.add_argument(
        "--corr-symbols",
        help="comma separated correlated symbol pairs e.g. EURUSD:USDCHF",
    )
    p.add_argument(
        "--calendar-file", help="CSV file with columns time,impact[,id] for events"
    )
    p.add_argument(
        "--event-window", type=float, default=60.0, help="minutes around events to flag"
    )
    args = p.parse_args()
    if args.corr_symbols:
        corr_map = {}
        for p_sym in args.corr_symbols.split(","):
            if ":" in p_sym:
                base, peer = p_sym.split(":", 1)
                corr_map.setdefault(base, []).append(peer)
    else:
        corr_map = None
    if args.calendar_file:
        events = _load_calendar(Path(args.calendar_file))
    else:
        events = None
    detect_regimes(
        args.data_dir,
        args.out_file,
        args.clusters,
        args.algorithm,
        args.model_json,
        args.assignments,
        corr_map=corr_map,
        calendar_events=events,
        event_window=args.event_window,
    )


if __name__ == "__main__":
    main()
