#!/usr/bin/env python3
"""Train a simple RL agent from trade logs."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction import DictVectorizer


# -------------------------------
# Data loading utilities
# -------------------------------

def _load_logs(data_dir: Path) -> List[Dict]:
    """Load raw log rows from ``data_dir``."""
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
        "comment",
    ]

    rows: List[Dict] = []
    for log_file in sorted(data_dir.glob("trades_*.csv")):
        with open(log_file, newline="") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                if len(row) == len(fields):
                    rows.append(dict(zip(fields, row)))
                else:
                    r = {fields[i]: row[i] for i in range(min(len(row), len(fields)))}
                    rows.append(r)
    return rows


def _pair_trades(rows: List[Dict]) -> List[Dict]:
    """Pair OPEN and CLOSE rows into trade records."""
    open_map: Dict[str, Dict] = {}
    trades: List[Dict] = []
    for r in rows:
        action = (r.get("action") or "").upper()
        ticket = r.get("ticket")
        if action == "OPEN":
            open_map[ticket] = r
        elif action == "CLOSE" and ticket in open_map:
            o = open_map.pop(ticket)
            profit = float(r.get("profit", 0) or 0)
            trades.append({"open": o, "profit": profit})
    return trades


def _extract_feature(row: Dict) -> Dict:
    """Extract feature dictionary from an OPEN row."""
    try:
        t = datetime.strptime(row["event_time"], "%Y.%m.%d %H:%M:%S")
    except ValueError:
        t = datetime.strptime(row["event_time"], "%Y.%m.%d %H:%M")

    price = float(row.get("price", 0) or 0)
    sl = float(row.get("sl", 0) or 0)
    tp = float(row.get("tp", 0) or 0)
    lots = float(row.get("lots", 0) or 0)

    return {
        "symbol": row.get("symbol", ""),
        "hour": t.hour,
        "lots": lots,
        "sl_dist": sl - price,
        "tp_dist": tp - price,
    }


# -------------------------------
# RL Training
# -------------------------------

def train(data_dir: Path, out_dir: Path) -> None:
    rows = _load_logs(data_dir)
    trades = _pair_trades(rows)

    if not trades:
        raise ValueError(f"No training data found in {data_dir}")

    feats: List[Dict] = []
    actions: List[int] = []
    rewards: List[float] = []
    for t in trades:
        o = t["open"]
        feats.append(_extract_feature(o))
        actions.append(0 if int(float(o.get("order_type", 0))) == 0 else 1)
        rewards.append(float(t["profit"]))

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform(feats)
    n_features = X.shape[1]

    weights = np.zeros((2, n_features))
    intercepts = np.zeros(2)
    alpha = 0.1

    for x, a, r in zip(X, actions, rewards):
        q_val = intercepts[a] + np.dot(weights[a], x)
        td_err = r - q_val
        weights[a] += alpha * td_err * x
        intercepts[a] += alpha * td_err

    preds = []
    for x in X:
        qb = intercepts[0] + np.dot(weights[0], x)
        qs = intercepts[1] + np.dot(weights[1], x)
        preds.append(0 if qb >= qs else 1)
    train_acc = float(np.mean(np.array(preds) == np.array(actions)))

    out_dir.mkdir(parents=True, exist_ok=True)
    model = {
        "model_id": "rl_agent",
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": vec.get_feature_names_out().tolist(),
        "coefficients": (weights[0] - weights[1]).tolist(),
        "intercept": float(intercepts[0] - intercepts[1]),
        "train_accuracy": train_acc,
        "val_accuracy": float("nan"),
        "accuracy": float("nan"),
        "num_samples": len(actions),
    }

    with open(out_dir / "model.json", "w") as f:
        json.dump(model, f, indent=2)

    print(f"Model written to {out_dir / 'model.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train RL agent from logs")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    train(Path(args.data_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
