#!/usr/bin/env python3
"""Train model from exported features.

The observer EA continuously exports trade logs as CSV files. This script
loads those logs, extracts a few simple features from each trade entry and
trains a very small predictive model. The resulting parameters along with
some training metadata are written to ``model.json`` so they can be consumed
by other helper scripts.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def _load_logs(data_dir: Path):
    """Load log rows from ``data_dir``.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``trades_*.csv`` files.

    Returns
    -------
    list[dict]
        Parsed rows as dictionaries.
    """

    fields = [
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "profit",
        "comment",
    ]

    rows = []
    for log_file in sorted(data_dir.glob("trades_*.csv")):
        with open(log_file, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            # If header is missing assume standard order
            for row in reader:
                if not row:
                    continue
                if len(row) == len(fields):
                    rows.append(dict(zip(fields, row)))
                else:
                    # best effort alignment
                    r = {
                        fields[i]: row[i]
                        for i in range(min(len(row), len(fields)))
                    }
                    rows.append(r)
    return rows


def _extract_features(rows):
    feature_dicts = []
    labels = []
    for r in rows:
        if r.get("action", "").upper() != "OPEN":
            continue

        try:
            t = datetime.strptime(r["event_time"], "%Y.%m.%d %H:%M:%S")
        except ValueError:
            try:
                t = datetime.strptime(r["event_time"], "%Y.%m.%d %H:%M")
            except Exception:
                continue

        order_type = int(float(r.get("order_type", 0)))
        label = 1 if order_type == 0 else 0  # buy=1, sell=0

        price = float(r.get("price", 0) or 0)
        sl = float(r.get("sl", 0) or 0)
        tp = float(r.get("tp", 0) or 0)
        lots = float(r.get("lots", 0) or 0)

        feat = {
            "symbol": r.get("symbol", ""),
            "hour": t.hour,
            "lots": lots,
            "sl_dist": sl - price,
            "tp_dist": tp - price,
        }

        feature_dicts.append(feat)
        labels.append(label)
    return feature_dicts, np.array(labels)


def train(data_dir: Path, out_dir: Path):
    """Train a simple logistic regression model from the log directory."""

    rows = _load_logs(data_dir)
    features, labels = _extract_features(rows)

    if not features:
        raise ValueError(f"No training data found in {data_dir}")

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(features)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, labels)

    preds = clf.predict(X)
    acc = float(accuracy_score(labels, preds))

    out_dir.mkdir(parents=True, exist_ok=True)

    model = {
        "model_id": "target_clone",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": vec.get_feature_names_out().tolist(),
        "coefficients": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "accuracy": acc,
        "num_samples": int(labels.shape[0]),
    }

    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f, indent=2)

    print(f"Model written to {out_dir / 'model.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()
    train(Path(args.data_dir), Path(args.out_dir))


if __name__ == '__main__':
    main()
