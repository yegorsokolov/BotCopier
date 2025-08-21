#!/usr/bin/env python3
"""Evaluate prediction accuracy from logs.

This utility compares a CSV file of predicted trade signals against the actual
trade log exported by ``Observer_TBot``.  Basic hit rate and profit metrics are
calculated and either printed to the console or written to a JSON report.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import math


def _parse_time(value: str) -> datetime:
    """Parse various timestamp formats used in logs."""

    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    raise ValueError(f"Unrecognised time format: {value}")


def _load_predictions(pred_file: Path) -> List[Dict]:
    """Load prediction rows from ``pred_file``."""

    with open(pred_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        preds = []
        for r in reader:
            ts = _parse_time(
                r.get("timestamp") or r.get("time") or r[reader.fieldnames[0]]
            )
            direction_raw = str(
                r.get("direction") or r.get("order_type") or ""
            ).strip().lower()

            if direction_raw in ("1", "buy"):
                direction = 1
            elif direction_raw in ("0", "-1", "sell"):
                direction = -1
            else:
                # default to buy when direction is missing
                direction = 1

            preds.append(
                {
                    "timestamp": ts,
                    "symbol": r.get("symbol", ""),
                    "direction": direction,
                    "lots": float(r.get("lots", 0) or 0),
                }
            )
    return preds


def _load_actual_trades(log_file: Path) -> List[Dict]:
    """Load completed trades from the observer trade log."""

    with open(log_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        open_map: Dict[str, Dict] = {}
        trades = []
        for r in reader:
            action = (r.get("action") or "").upper()
            ticket = r.get("ticket")
            ts = _parse_time(
                r.get("event_time")
                or r.get("time_event")
                or r[reader.fieldnames[0]]
            )
            if action == "OPEN":
                open_map[ticket] = {
                    "open_time": ts,
                    "symbol": r.get("symbol", ""),
                    "direction": 1
                    if int(float(r.get("order_type", 0))) == 0
                    else -1,
                    "lots": float(r.get("lots", 0) or 0),
                }
            elif action == "CLOSE" and ticket in open_map:
                o = open_map.pop(ticket)
                profit = float(r.get("profit", 0) or 0)
                trade = {
                    **o,
                    "close_time": ts,
                    "profit": profit,
                }
                trades.append(trade)
    return trades


def evaluate(pred_file: Path, actual_log: Path, window: int) -> Dict:
    """Compare predictions to actual trades and compute summary statistics."""

    predictions = _load_predictions(pred_file)
    actual_trades = _load_actual_trades(actual_log)

    matches = 0
    gross_profit = 0.0
    gross_loss = 0.0
    profits: List[float] = []
    used = set()
    for pred in predictions:
        best = None
        for idx, trade in enumerate(actual_trades):
            if idx in used:
                continue
            if trade["symbol"] != pred["symbol"]:
                continue
            if trade["direction"] != pred["direction"]:
                continue
            delta = (trade["open_time"] - pred["timestamp"]).total_seconds()
            if 0 <= delta <= window:
                best = idx
                break

        if best is not None:
            used.add(best)
            matches += 1
            p = actual_trades[best]["profit"]
            profits.append(p)
            if p >= 0:
                gross_profit += p
            else:
                gross_loss += -p

    tp = matches
    fp = len(predictions) - matches
    fn = len(actual_trades) - matches
    total = tp + fp + fn

    accuracy = tp / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    profit_factor = (gross_profit / gross_loss) if gross_loss else float("inf")

    # Basic profit statistics
    expected_return = sum(profits) / len(profits) if profits else 0.0
    downside = [p for p in profits if p < 0]
    downside_risk = -sum(downside) / len(downside) if downside else 0.0

    # Sharpe ratio uses overall standard deviation while Sortino only
    # considers downside volatility.  Both expect non-zero deviation.
    sharpe = 0.0
    sortino = 0.0
    if len(profits) > 1:
        mean = expected_return
        variance = sum((p - mean) ** 2 for p in profits) / (len(profits) - 1)
        std = math.sqrt(variance)
        if std > 0:
            sharpe = mean / std

        downside_dev = 0.0
        if downside:
            downside_dev = math.sqrt(
                sum((p) ** 2 for p in downside) / len(downside)
            )
        if downside_dev > 0:
            sortino = mean / downside_dev

    # Expectancy expresses the average profit per trade taking win/loss
    # probabilities into account.
    win_profits = [p for p in profits if p > 0]
    loss_profits = [p for p in profits if p < 0]
    win_rate = len(win_profits) / len(profits) if profits else 0.0
    avg_win = (sum(win_profits) / len(win_profits)) if win_profits else 0.0
    avg_loss = (-sum(loss_profits) / len(loss_profits)) if loss_profits else 0.0
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    return {
        "predicted_events": len(predictions),
        "actual_events": len(actual_trades),
        "matched_events": matches,
        "hit_rate": precision,
        "coverage": recall,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "expectancy": expectancy,
        "expected_return": expected_return,
        "downside_risk": downside_risk,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate prediction logs")
    p.add_argument("predicted_log", help="CSV of predicted signals")
    p.add_argument("actual_log", help="CSV of observed trades")
    p.add_argument(
        "--window",
        type=int,
        default=60,
        help="matching window in seconds",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        help="optional path for JSON summary",
    )
    args = p.parse_args()

    stats = evaluate(
        Path(args.predicted_log),
        Path(args.actual_log),
        args.window,
    )

    out_path = args.json_out or Path(args.predicted_log).parent / "evaluation.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote report to {out_path}")

    print("--- Evaluation Summary ---")
    print(f"Predicted events : {stats['predicted_events']}")
    print(
        f"Matched events   : {stats['matched_events']}"
        f" ({stats['precision']*100:.1f}% precision)"
    )
    print(f"Recall           : {stats['recall']*100:.1f}% of actual trades")
    print(f"Accuracy         : {stats['accuracy']*100:.1f}%")
    print(
        f"Profit Factor    : {stats['profit_factor']:.2f}"
        f" (gross P/L: {stats['gross_profit']-stats['gross_loss']:.2f})"
    )
    print(f"Sharpe Ratio     : {stats['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio    : {stats['sortino_ratio']:.2f}")
    print(f"Expectancy       : {stats['expectancy']:.2f}")


if __name__ == '__main__':
    main()
