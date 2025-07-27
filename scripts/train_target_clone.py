#!/usr/bin/env python3
"""Train model from exported features.

The observer EA continuously exports trade logs as CSV files. This script
loads those logs, extracts a few simple features from each trade entry and
trains a very small predictive model. The resulting parameters along with
some training metadata are written to ``model.json`` so they can be consumed
by other helper scripts.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    HAS_TF = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TF = False

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV


def _sma(values, window):
    """Simple moving average for the last ``window`` values."""
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(sum(values[-w:]) / w)


def _atr(values, period):
    """Average true range over ``period`` price changes."""
    if len(values) < 2:
        return 0.0
    diffs = np.abs(np.diff(values[-(period + 1) :]))
    if len(diffs) == 0:
        return 0.0
    w = min(period, len(diffs))
    return float(diffs[-w:].mean())


def _bollinger(values, window, dev=2.0):
    """Return Bollinger Bands for the last ``window`` values."""
    if not values:
        return 0.0, 0.0, 0.0
    w = min(window, len(values))
    arr = np.array(values[-w:])
    sma = arr.mean()
    std = arr.std(ddof=0)
    upper = sma + dev * std
    lower = sma - dev * std
    return float(upper), float(sma), float(lower)


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


def _load_logs(data_dir: Path) -> pd.DataFrame:
    """Load log rows from ``data_dir``.

    ``MODIFY`` entries are retained alongside ``OPEN`` and ``CLOSE``.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``trades_*.csv`` files.

    Returns
    -------
    pandas.DataFrame
        Parsed rows as a DataFrame.
    """

    fields = [
        "event_id",
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
        "spread",
        "comment",
        "remaining_lots",
    ]

    dfs: List[pd.DataFrame] = []
    for log_file in sorted(data_dir.glob("trades_*.csv")):
        df = pd.read_csv(
            log_file,
            sep=";",
            names=fields,
            header=0,
            parse_dates=["event_time"],
        )
        dfs.append(df)

    if dfs:
        df_logs = pd.concat(dfs, ignore_index=True)
    else:
        df_logs = pd.DataFrame(columns=fields)

    df_logs.columns = [c.lower() for c in df_logs.columns]

    valid_actions = {"OPEN", "CLOSE", "MODIFY"}
    df_logs["action"] = df_logs["action"].fillna("").str.upper()
    df_logs = df_logs[(df_logs["action"] == "") | df_logs["action"].isin(valid_actions)]

    metrics_file = data_dir / "metrics.csv"
    if metrics_file.exists():
        df_metrics = pd.read_csv(metrics_file, sep=";")
        df_metrics.columns = [c.lower() for c in df_metrics.columns]
        if "magic" in df_metrics.columns:
            key_col = "magic"
        elif "model_id" in df_metrics.columns:
            key_col = "model_id"
        else:
            key_col = None

        if key_col is not None:
            df_metrics[key_col] = pd.to_numeric(df_metrics[key_col], errors="coerce").fillna(0).astype(int)
            df_logs["magic"] = pd.to_numeric(df_logs["magic"], errors="coerce").fillna(0).astype(int)
            if key_col == "magic":
                df_logs = df_logs.merge(df_metrics, how="left", on="magic")
            else:
                df_logs = df_logs.merge(df_metrics, how="left", left_on="magic", right_on="model_id")
                df_logs = df_logs.drop(columns=["model_id"])

    return df_logs


def _extract_features(
    rows,
    use_sma=False,
    sma_window=5,
    use_rsi=False,
    rsi_period=14,
    use_macd=False,
    use_atr=False,
    atr_period=14,
    use_bollinger=False,
    boll_window=20,
    volatility=None,
):
    feature_dicts = []
    labels = []
    prices = []
    macd_state = {}
    for r in rows:
        if r.get("action", "").upper() != "OPEN":
            continue

        t = r["event_time"]
        if not isinstance(t, datetime):
            parsed = None
            for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
                try:
                    parsed = datetime.strptime(str(t), fmt)
                    break
                except Exception:
                    continue
            if parsed is None:
                continue
            t = parsed

        order_type = int(float(r.get("order_type", 0)))
        label = 1 if order_type == 0 else 0  # buy=1, sell=0

        price = float(r.get("price", 0) or 0)
        sl = float(r.get("sl", 0) or 0)
        tp = float(r.get("tp", 0) or 0)
        lots = float(r.get("lots", 0) or 0)

        spread = float(r.get("spread", 0) or 0)

        feat = {
            "symbol": r.get("symbol", ""),
            "hour": t.hour,
            "day_of_week": t.weekday(),
            "lots": lots,
            "sl_dist": sl - price,
            "tp_dist": tp - price,
            "spread": spread,
        }

        if volatility is not None:
            key = t.strftime("%Y-%m-%d %H")
            vol = volatility.get(key)
            if vol is None:
                key = t.strftime("%Y-%m-%d")
                vol = volatility.get(key, 0.0)
            feat["volatility"] = float(vol)

        if use_sma:
            feat["sma"] = _sma(prices, sma_window)

        if use_rsi:
            feat["rsi"] = _rsi(prices, rsi_period)

        if use_macd:
            macd, signal = _macd_update(macd_state, price)
            feat["macd"] = macd
            feat["macd_signal"] = signal

        if use_atr:
            feat["atr"] = _atr(prices, atr_period)

        if use_bollinger:
            upper, mid, lower = _bollinger(prices, boll_window)
            feat["bollinger_upper"] = upper
            feat["bollinger_middle"] = mid
            feat["bollinger_lower"] = lower

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
    use_atr: bool = False,
    atr_period: int = 14,
    use_bollinger: bool = False,
    boll_window: int = 20,
    volatility_series=None,
    grid_search: bool = False,
    c_values=None,
    model_type: str = "logreg",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    incremental: bool = False,
    sequence_length: int = 5,
):
    """Train a simple classifier model from the log directory."""

    rows_df = _load_logs(data_dir)
    features, labels = _extract_features(
        rows_df.to_dict("records"),
        use_sma=use_sma,
        sma_window=sma_window,
        use_rsi=use_rsi,
        rsi_period=rsi_period,
        use_macd=use_macd,
        use_atr=use_atr,
        atr_period=atr_period,
        use_bollinger=use_bollinger,
        boll_window=boll_window,
        volatility=volatility_series,
    )

    if not features:
        raise ValueError(f"No training data found in {data_dir}")

    existing_model = None
    if incremental:
        model_file = out_dir / "model.json"
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} not found for incremental training")
        with open(model_file) as f:
            existing_model = json.load(f)

    if existing_model is not None:
        vec = DictVectorizer(sparse=False)
        vec.fit([{name: 0.0} for name in existing_model.get("feature_names", [])])
    else:
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

    if existing_model is not None:
        X_train = vec.transform(feat_train)
    else:
        X_train = vec.fit_transform(feat_train)
    if feat_val:
        X_val = vec.transform(feat_val)
    else:
        X_val = np.empty((0, X_train.shape[1]))

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        train_proba = clf.predict_proba(X_train)[:, 1]
        val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
    elif model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed")
        clf = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        clf.fit(X_train, y_train)
        train_proba = clf.predict_proba(X_train)[:, 1]
        val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
    elif model_type == "nn":
        if HAS_TF:
            model_nn = keras.Sequential([
                keras.layers.Input(shape=(X_train.shape[1],)),
                keras.layers.Dense(8, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ])
            model_nn.compile(optimizer="adam", loss="binary_crossentropy")
            model_nn.fit(X_train, y_train, epochs=50, verbose=0)
            train_proba = model_nn.predict(X_train).reshape(-1)
            val_proba = (
                model_nn.predict(X_val).reshape(-1) if len(y_val) > 0 else np.empty(0)
            )
            clf = model_nn
        else:
            clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42)
            clf.fit(X_train, y_train)
            train_proba = clf.predict_proba(X_train)[:, 1]
            val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)
    elif model_type == "lstm":
        if not HAS_TF:
            raise ImportError("TensorFlow is required for LSTM model")
        seq_len = sequence_length
        X_all = vec.fit_transform(features) if existing_model is None else vec.transform(features)
        sequences = []
        for i in range(len(X_all)):
            start = max(0, i - seq_len + 1)
            seq = X_all[start : i + 1]
            if seq.shape[0] < seq_len:
                pad = np.zeros((seq_len - seq.shape[0], X_all.shape[1]))
                seq = np.vstack([pad, seq])
            sequences.append(seq)
        X_all_seq = np.array(sequences)
        if len(labels) < 5 or len(np.unique(labels)) < 2:
            X_train_seq, y_train = X_all_seq, labels
            X_val_seq, y_val = np.empty((0, seq_len, X_all.shape[1])), np.array([])
        else:
            X_train_seq, X_val_seq, y_train, y_val = train_test_split(
                X_all_seq,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels,
            )
        model_nn = keras.Sequential([
            keras.layers.Input(shape=(seq_len, X_all.shape[1])),
            keras.layers.LSTM(8),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        model_nn.compile(optimizer="adam", loss="binary_crossentropy")
        model_nn.fit(X_train_seq, y_train, epochs=50, verbose=0)
        train_proba = model_nn.predict(X_train_seq).reshape(-1)
        val_proba = model_nn.predict(X_val_seq).reshape(-1) if len(y_val) > 0 else np.empty(0)
        clf = model_nn
    else:
        if grid_search:
            if c_values is None:
                c_values = [0.01, 0.1, 1.0, 10.0]
            param_grid = {"C": c_values}
            gs = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=3)
            gs.fit(X_train, y_train)
            clf = gs.best_estimator_
        else:
            clf = LogisticRegression(max_iter=200, warm_start=existing_model is not None)
            if existing_model is not None:
                clf.classes_ = np.array([0, 1])
                clf.coef_ = np.array([existing_model.get("coefficients", [])])
                clf.intercept_ = np.array([existing_model.get("intercept", 0.0)])
            clf.fit(X_train, y_train)
            train_proba = clf.predict_proba(X_train)[:, 1]
            val_proba = clf.predict_proba(X_val)[:, 1] if len(y_val) > 0 else np.empty(0)

    if len(y_val) > 0:
        threshold, _ = _best_threshold(y_val, val_proba)
        val_preds = (val_proba >= threshold).astype(int)
        val_acc = float(accuracy_score(y_val, val_preds))
    else:
        threshold = 0.5
        val_acc = float("nan")
    train_preds = (train_proba >= threshold).astype(int)
    train_acc = float(accuracy_score(y_train, train_preds))

    # Compute SHAP feature importance on the training set
    try:
        import shap  # type: ignore

        explainer = shap.Explainer(clf, X_train)
        shap_values = explainer(X_train)
        importances = np.abs(shap_values.values).mean(axis=0)
        feature_importance = dict(
            zip(vec.get_feature_names_out().tolist(), importances.tolist())
        )
    except Exception:  # pragma: no cover - shap optional
        feature_importance = {}

    out_dir.mkdir(parents=True, exist_ok=True)

    model = {
        "model_id": (existing_model.get("model_id") if existing_model else "target_clone"),
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": vec.get_feature_names_out().tolist(),
        "model_type": model_type,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "threshold": threshold,
        # main accuracy metric is validation performance when available
        "accuracy": val_acc,
        "num_samples": int(labels.shape[0]) + (int(existing_model.get("num_samples", 0)) if existing_model else 0),
        "feature_importance": feature_importance,
    }

    if model_type == "logreg":
        model["coefficients"] = clf.coef_[0].tolist()
        model["intercept"] = float(clf.intercept_[0])
    elif model_type == "xgboost":
        # approximate tree ensemble with linear model for MQL4 export
        logit_p = np.log(train_proba / (1.0 - train_proba + 1e-9))
        A = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        coef = np.linalg.lstsq(A, logit_p, rcond=None)[0]
        model["coefficients"] = coef[1:].tolist()
        model["intercept"] = float(coef[0])

        # lookup probabilities per trading hour for simple export
        feature_names = vec.get_feature_names_out().tolist()
        base_feat = {name: 0.0 for name in feature_names}
        lookup = []
        for h in range(24):
            f = base_feat.copy()
            if "hour" in f:
                f["hour"] = float(h)
            X_h = vec.transform([f])
            lookup.append(float(clf.predict_proba(X_h)[0, 1]))
        model["probability_table"] = lookup
    elif model_type == "nn":
        if HAS_TF:
            weights = [w.tolist() for w in clf.get_weights()]
        else:
            weights = [
                clf.coefs_[0].tolist(),
                clf.intercepts_[0].tolist(),
                clf.coefs_[1].tolist(),
                clf.intercepts_[1].tolist(),
            ]
        model["nn_weights"] = weights
        model["hidden_size"] = len(weights[1]) if weights else 0
    elif model_type == "lstm":
        weights = [w.tolist() for w in clf.get_weights()]
        model["lstm_weights"] = weights
        model["sequence_length"] = sequence_length
        model["hidden_size"] = len(weights[1]) // 4 if weights else 0

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
    p.add_argument('--use-atr', action='store_true', help='include ATR feature')
    p.add_argument('--use-bollinger', action='store_true', help='include Bollinger Bands feature')
    p.add_argument('--volatility-file', help='JSON file with precomputed volatility')
    p.add_argument('--grid-search', action='store_true', help='enable grid search with cross-validation')
    p.add_argument('--c-values', type=float, nargs='*')
    p.add_argument('--model-type', choices=['logreg', 'random_forest', 'xgboost', 'nn', 'lstm'], default='logreg',
                   help='classifier type')
    p.add_argument('--sequence-length', type=int, default=5, help='LSTM sequence length')
    p.add_argument('--n-estimators', type=int, default=100, help='xgboost trees')
    p.add_argument('--learning-rate', type=float, default=0.1, help='xgboost learning rate')
    p.add_argument('--max-depth', type=int, default=3, help='xgboost tree depth')
    p.add_argument('--incremental', action='store_true', help='update existing model.json')
    args = p.parse_args()
    if args.volatility_file:
        import json
        with open(args.volatility_file) as f:
            vol_data = json.load(f)
    else:
        vol_data = None
    train(
        Path(args.data_dir),
        Path(args.out_dir),
        use_sma=args.use_sma,
        sma_window=args.sma_window,
        use_rsi=args.use_rsi,
        rsi_period=args.rsi_period,
        use_macd=args.use_macd,
        use_atr=args.use_atr,
        use_bollinger=args.use_bollinger,
        volatility_series=vol_data,
        grid_search=args.grid_search,
        c_values=args.c_values,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        incremental=args.incremental,
        sequence_length=args.sequence_length,
    )


if __name__ == '__main__':
    main()
