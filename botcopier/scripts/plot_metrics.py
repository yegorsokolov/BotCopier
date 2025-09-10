#!/usr/bin/env python3
"""Plot metrics over time from metrics.csv.

Includes graphs for file write and socket send error counters emitted by the
trading observer.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def _load_metrics(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            try:
                ts = datetime.strptime(r["time"], "%Y.%m.%d %H:%M")
            except Exception:
                continue
            try:
                magic = int(float(r.get("magic") or r.get("model_id") or 0))
            except Exception:
                magic = 0
            rows.append(
                {
                    "time": ts,
                    "magic": magic,
                    "win_rate": float(r.get("win_rate", 0) or 0),
                    "avg_profit": float(r.get("avg_profit", 0) or 0),
                    "trade_count": int(float(r.get("trade_count", 0) or 0)),
                    "drawdown": float(r.get("drawdown", 0) or 0),
                    "sharpe": float(r.get("sharpe", 0) or 0),
                    "sortino": float(r.get("sortino", 0) or 0),
                    "expectancy": float(r.get("expectancy", 0) or 0),
                    "cvar": float(r.get("cvar", 0) or 0),
                    "file_write_errors": int(
                        float(r.get("file_write_errors") or r.get("write_errors") or 0)
                    ),
                    "socket_errors": int(float(r.get("socket_errors", 0) or 0)),
                    "var_breach_count": int(float(r.get("var_breach_count", 0) or 0)),
                    "risk_weight": float(r.get("risk_weight", 0) or 0),
                }
            )
    return rows


def _plot(rows, magic=None):
    if magic is not None:
        rows = [r for r in rows if r["magic"] == magic]
    if not rows:
        print("No matching rows")
        return
    rows.sort(key=lambda r: r["time"])
    times = [r["time"] for r in rows]
    win_rate = [r["win_rate"] for r in rows]
    drawdown = [r["drawdown"] for r in rows]
    sharpe = [r["sharpe"] for r in rows]
    sortino = [r.get("sortino", 0) for r in rows]
    expectancy = [r.get("expectancy", 0) for r in rows]
    write_err = [r["file_write_errors"] for r in rows]
    socket_err = [r["socket_errors"] for r in rows]
    var_breach = [r.get("var_breach_count", 0) for r in rows]
    risk_w = [r.get("risk_weight", 0) for r in rows]
    cvar = [r.get("cvar", 0) for r in rows]

    fig, (ax1, ax3, ax4) = plt.subplots(3, 1, sharex=True)
    ax1.plot(times, win_rate, label="Win Rate")
    ax1.set_ylabel("Win Rate")

    ax2 = ax1.twinx()
    ax2.plot(times, drawdown, color="r", label="Drawdown")
    ax2.plot(times, sharpe, color="g", label="Sharpe")
    ax2.plot(times, sortino, color="c", label="Sortino")
    ax2.set_ylabel("Drawdown / Ratios")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax3.plot(times, write_err, label="File Write Errors", color="purple")
    ax3.plot(times, socket_err, label="Socket Errors", color="orange")
    ax3.plot(times, var_breach, label="VaR Breaches", color="red")
    if any(expectancy):
        ax3b = ax3.twinx()
        ax3b.plot(times, expectancy, label="Expectancy", color="brown")
        ax3b.set_ylabel("Expectancy")
        ax3b.legend(loc="upper right")
    ax3.legend(loc="upper left")
    ax3.set_ylabel("Error Count")

    ax4.plot(times, risk_w, label="Risk Weight", color="black")
    ax4.set_ylabel("Risk Weight")
    if any(cvar):
        ax4b = ax4.twinx()
        ax4b.plot(times, cvar, label="CVaR", color="red")
        ax4b.set_ylabel("CVaR")
        ax4b.legend(loc="upper right")
    ax4.legend(loc="upper left")
    ax4.set_xlabel("Time")
    plt.tight_layout()
    plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Plot metrics over time")
    p.add_argument("metrics_file", help="Path to metrics.csv")
    p.add_argument("--magic", type=int, help="Filter by magic/model id")
    args = p.parse_args()

    rows = _load_metrics(Path(args.metrics_file))
    _plot(rows, args.magic)


if __name__ == "__main__":
    main()
