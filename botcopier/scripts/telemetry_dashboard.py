#!/usr/bin/env python3
"""Visualize correlations between system telemetry and trading metrics."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def _load_csv(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Plot telemetry vs metrics correlations")
    p.add_argument("--metrics", default="logs/metrics.csv", help="Path to metrics log")
    p.add_argument(
        "--telemetry",
        default="logs/system_telemetry.csv",
        help="Path to system telemetry log",
    )
    args = p.parse_args()

    metrics = {r.get("trace_id"): r for r in _load_csv(Path(args.metrics))}
    telem = _load_csv(Path(args.telemetry))

    rows = []
    for t in telem:
        m = metrics.get(t.get("trace_id"))
        if not m:
            continue
        try:
            ts = datetime.fromisoformat(t["time"])
        except Exception:
            ts = None
        rows.append(
            {
                "time": ts,
                "cpu": float(t.get("cpu_percent", 0) or 0),
                "mem": float(t.get("mem_percent", 0) or 0),
                "socket_errors": int(m.get("socket_errors", 0) or 0),
            }
        )

    if not rows:
        print("No overlapping telemetry and metric records")
        return

    rows.sort(key=lambda r: r["time"] or datetime.min)
    times = [r["time"] for r in rows]
    cpu = [r["cpu"] for r in rows]
    sock = [r["socket_errors"] for r in rows]

    fig, ax1 = plt.subplots()
    ax1.plot(times, cpu, label="CPU %")
    ax1.set_ylabel("CPU %")

    ax2 = ax1.twinx()
    ax2.plot(times, sock, color="r", label="Socket Errors")
    ax2.set_ylabel("Socket Errors")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title("CPU usage vs Socket Errors")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
