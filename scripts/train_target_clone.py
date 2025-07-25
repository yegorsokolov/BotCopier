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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV


def _sma(values, window):
    """Simple moving average for the last ``window`` values."""
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(sum(values[-w:]) / w)


def _rsi(values, period):
    """Very small RSI implementation on ``values``."""
    if len(values) < 2:
        return 50.0
    deltas = np.diff(values[-(period + 1) :])
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    if losses == 0:
        return 100.0
    rs = gains / losses
    return float(100 - (100 / (1 + rs)))


def _macd_update(state, price, short=12, long=26, signal=9):
    """Update MACD EMA state with ``price`` and return (macd, signal)."""
    alpha_short = 2 / (short + 1)
    alpha_long = 2 / (long + 1)
    alpha_signal = 2 / (signal + 1)

    ema_short = state.get("ema_short")
    ema_long = state.get("ema_long")
    ema_signal = state.get("ema_signal")

    ema_short = price if ema_short is None else alpha_short * price + (1 - alpha_short) * ema_short
    ema_long = price if ema_long is None else alpha_long * price + (1 - alpha_long) * ema_long
    macd = ema_short - ema_long
    ema_signal = macd if ema_signal is None else alpha_signal * macd + (1 - alpha_signal) * ema_signal

    state["ema_short"] = ema_short
    state["ema_long"] = ema_long
    state["ema_signal"] = ema_signal

    return macd, ema_signal


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

    # Attach metrics if available
    metrics_file = data_dir / "metrics.csv"
    metrics_map = {}
    if metrics_file.exists():
        with open(metrics_file, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            header_l = [h.lower() for h in header] if header else []
            if header_l:
                try:
                    m_idx = header_l.index("magic") if "magic" in header_l else header_l.index("model_id")
                except ValueError:
                    m_idx = None
            else:
                m_idx = None

            if m_idx is not None:
                for row in reader:
                    if not row:
                        continue
                    try:
                        key = row[m_idx] if m_idx < len(row) else "0"
                        magic = int(float(key or 0))
                    except Exception:
                        magic = 0
                    metrics_map[magic] = {
                        header_l[i]: row[i]
                        for i in range(min(len(header_l), len(row)))
                    }

    for r in rows:
        try:
            magic = int(float(r.get("magic", 0) or 0))
        except Exception:
            magic = 0
        m = metrics_map.get(magic)
        if m:
            for k, v in m.items():
                if k in ("magic", "model_id"):
                    continue
                r[k] = v

    return rows


def _extract_features(
    rows,
    use_sma=False,
    sma_window=5,
    use_rsi=False,
    rsi_period=14,
    use_macd=False,
):
    feature_dicts = []
    labels = []
    prices = []
    macd_state = {}
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

        if use_sma:
            feat["sma"] = _sma(prices, sma_window)

        if use_rsi:
            feat["rsi"] = _rsi(prices, rsi_period)

        if use_macd:
            macd, signal = _macd_update(macd_state, price)
            feat["macd"] = macd
            feat["macd_signal"] = signal

        prices.append(price)

        feature_dicts.append(feat)
        labels.append(label)
    return feature_dicts, np.array(labels)


def _best_threshold(y_true, probas):
    """Return probability threshold with best F1 score."""
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.1, 0.9, 17):
        preds = (probas >= t).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, best_f1


def train(
    data_dir: Path,
    out_dir: Path,
    *,
    use_sma: bool = False,
    sma_window: int = 5,
    use_rsi: bool = False,
    rsi_period: int = 14,
    use_macd: bool = False,
    grid_search: bool = False,
    c_values=None,
    model_type: str = "logreg",
):
    """Train a simple classifier model from the log directory."""

    rows = _load_logs(data_dir)
    features, labels = _extract_features(
        rows,
        use_sma=use_sma,
        sma_window=sma_window,
        use_rsi=use_rsi,
        rsi_period=rsi_period,
        use_macd=use_macd,
    )

    if not features:
        raise ValueError(f"No training data found in {data_dir}")

    vec = DictVectorizer(sparse=False)

    if len(labels) < 5 or len(np.unique(labels)) < 2:
        # Not enough data to create a meaningful split
        feat_train, y_train = features, labels
        feat_val, y_val = [], np.array([])
    else:
        feat_train, feat_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        # if the training split ended up with only one class, fall back to using
        # all data for training so the model can be fit
        if len(np.unique(y_train)) < 2:
            feat_train, y_train = features, labels
            feat_val, y_val = [], np.array([])

    X_train = vec.fit_transform(feat_train)
    if feat_val:
        X_val = vec.transform(feat_val)
    else:
        X_val = np.empty((0, X_train.shape[1]))

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
    else:
        if grid_search:
            if c_values is None:
                c_values = [0.01, 0.1, 1.0, 10.0]
            param_grid = {"C": c_values}
            gs = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=3)
            gs.fit(X_train, y_train)
            clf = gs.best_estimator_
        else:
            clf = LogisticRegression(max_iter=200)
            clf.fit(X_train, y_train)

    train_proba = clf.predict_proba(X_train)[:, 1]
    if len(y_val) > 0:
        val_proba = clf.predict_proba(X_val)[:, 1]
        threshold, _ = _best_threshold(y_val, val_proba)
        val_preds = (val_proba >= threshold).astype(int)
        val_acc = float(accuracy_score(y_val, val_preds))
    else:
        threshold = 0.5
        val_acc = float("nan")
    train_preds = (train_proba >= threshold).astype(int)
    train_acc = float(accuracy_score(y_train, train_preds))

    out_dir.mkdir(parents=True, exist_ok=True)

    model = {
        "model_id": "target_clone",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": vec.get_feature_names_out().tolist(),
        "model_type": model_type,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "threshold": threshold,
        # main accuracy metric is validation performance when available
        "accuracy": val_acc,
        "num_samples": int(labels.shape[0]),
    }

    if model_type == "logreg":
        model["coefficients"] = clf.coef_[0].tolist()
        model["intercept"] = float(clf.intercept_[0])

    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f, indent=2)

    print(f"Model written to {out_dir / 'model.json'}")
    print(f"Validation accuracy: {val_acc:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--use-sma', action='store_true', help='include moving average feature')
    p.add_argument('--sma-window', type=int, default=5)
    p.add_argument('--use-rsi', action='store_true', help='include RSI feature')
    p.add_argument('--rsi-period', type=int, default=14)
    p.add_argument('--use-macd', action='store_true', help='include MACD feature')
    p.add_argument('--grid-search', action='store_true', help='enable grid search with cross-validation')
    p.add_argument('--c-values', type=float, nargs='*')
    p.add_argument('--model-type', choices=['logreg', 'random_forest'], default='logreg',
                   help='classifier type')
    args = p.parse_args()
    train(
        Path(args.data_dir),
        Path(args.out_dir),
        use_sma=args.use_sma,
        sma_window=args.sma_window,
        use_rsi=args.use_rsi,
        rsi_period=args.rsi_period,
        use_macd=args.use_macd,
        grid_search=args.grid_search,
        c_values=args.c_values,
        model_type=args.model_type,
    )


if __name__ == '__main__':
    main()
