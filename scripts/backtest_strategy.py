#!/usr/bin/env python3
"""Backtest a generated MQ4 strategy using historical tick data.

This utility parses basic parameters from a generated MQ4 file (or JSON
parameter file) and simulates trades over a tick series.  Results including
win rate, profit factor, drawdown and Sharpe ratio are written to an output
report.  Optionally the metrics are appended to ``metrics.csv`` for
comparison with live trading.
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


@dataclass
class Tick:
    time: str
    bid: float
    ask: float


def load_strategy_params(path: Path) -> Dict[str, float]:
    """Extract parameters from a generated MQ4 or JSON file."""
    text = path.read_text()
    params: Dict[str, float] = {}
    m = re.search(r"MagicNumber\s*=\s*(\d+)", text)
    if m:
        params["magic"] = int(m.group(1))
    m = re.search(r"ModelThreshold\s*=\s*([0-9eE+\-.]+)", text)
    if m:
        try:
            params["threshold"] = float(m.group(1))
        except ValueError:
            pass
    # if JSON content (from model.json) is provided
    if path.suffix.lower() == ".json":
        try:
            obj = json.loads(text)
            if "magic" in obj:
                params.setdefault("magic", int(obj["magic"]))
            if "threshold" in obj:
                params.setdefault("threshold", float(obj["threshold"]))
        except json.JSONDecodeError:
            pass
    return params


def load_ticks(file: Path) -> List[Tick]:
    """Load semicolon separated tick data."""
    rows: List[Tick] = []
    with open(file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            rows.append(
                Tick(
                    time=r.get("time", ""),
                    bid=float(r.get("bid", 0) or 0),
                    ask=float(r.get("ask", 0) or 0),
                )
            )
    return rows


def backtest(ticks: List[Tick], threshold: float = 0.0) -> Dict[str, float]:
    """Run a very small backtest using price changes between ticks."""
    if len(ticks) < 2:
        return {"trade_count": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "drawdown": 0.0, "sharpe": 0.0, "avg_profit": 0.0}

    trades: List[float] = []
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    prev_price = ticks[0].bid

    for t in ticks[1:]:
        change = t.bid - prev_price
        if abs(change) < threshold:
            prev_price = t.bid
            continue
        profit = change
        trades.append(profit)
        equity += profit
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
        prev_price = t.bid

    trade_count = len(trades)
    if trade_count == 0:
        return {"trade_count": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "drawdown": 0.0, "sharpe": 0.0, "avg_profit": 0.0}

    wins = sum(1 for p in trades if p > 0)
    gross_profit = sum(p for p in trades if p > 0)
    gross_loss = -sum(p for p in trades if p < 0)
    win_rate = wins / trade_count
    profit_factor = (gross_profit / gross_loss) if gross_loss else float("inf")
    avg_profit = mean(trades)
    sd = pstdev(trades) if trade_count > 1 else 0.0
    sharpe = (avg_profit / sd * trade_count ** 0.5) if sd > 0 else 0.0
    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "drawdown": max_dd,
        "sharpe": sharpe,
        "avg_profit": avg_profit,
    }


def write_report(metrics: Dict[str, float], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)


def update_metrics_csv(metrics: Dict[str, float], metrics_file: Path, magic: int) -> None:
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    header = ["time", "magic", "win_rate", "avg_profit", "trade_count", "drawdown", "sharpe"]
    exists = metrics_file.exists()
    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        if not exists:
            writer.writerow(header)
        writer.writerow([
            datetime.utcnow().strftime("%Y.%m.%d %H:%M"),
            str(magic),
            f"{metrics['win_rate']:.6f}",
            f"{metrics['avg_profit']:.6f}",
            str(metrics['trade_count']),
            f"{metrics['drawdown']:.6f}",
            f"{metrics['sharpe']:.6f}",
        ])


def run_backtest(params_file: Path, tick_file: Path) -> Dict[str, float]:
    params = load_strategy_params(params_file)
    ticks = load_ticks(tick_file)
    result = backtest(ticks, params.get("threshold", 0.0))
    result["magic"] = params.get("magic", 0)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest a generated MQ4 strategy")
    p.add_argument("params_file", help="Path to MQ4 or JSON parameter file")
    p.add_argument("tick_file", help="CSV file of tick data")
    p.add_argument("--report", required=True, help="Output report JSON file")
    p.add_argument("--metrics-file", help="metrics.csv to append results to")
    args = p.parse_args()

    result = run_backtest(Path(args.params_file), Path(args.tick_file))
    write_report(result, Path(args.report))
    if args.metrics_file:
        update_metrics_csv(result, Path(args.metrics_file), result.get("magic", 0))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
