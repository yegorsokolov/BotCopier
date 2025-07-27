#!/usr/bin/env python3
"""Compute basic metrics from exported tick history."""
import argparse
import csv
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import statistics


def load_ticks(file: Path) -> List[Dict]:
    with open(file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = []
        for r in reader:
            rows.append(
                {
                    "time": r.get("time"),
                    "bid": float(r.get("bid", 0) or 0),
                    "ask": float(r.get("ask", 0) or 0),
                }
            )
        return rows


def compute_metrics(rows: List[Dict]) -> Dict:
    if not rows:
        return {"tick_count": 0, "avg_spread": 0.0, "price_change": 0.0}
    spreads = [r["ask"] - r["bid"] for r in rows]
    avg_spread = sum(spreads) / len(spreads)
    price_change = rows[-1]["bid"] - rows[0]["bid"]
    return {
        "tick_count": len(rows),
        "avg_spread": avg_spread,
        "price_change": price_change,
    }


def compute_volatility(rows: List[Dict], interval: str = "hourly") -> Dict[str, float]:
    """Return standard deviation of tick returns per interval.

    Parameters
    ----------
    rows : list of dict
        Tick rows as returned by :func:`load_ticks`.
    interval : str, optional
        ``"hourly"`` or ``"daily"`` grouping. Default is hourly.
    """

    if len(rows) < 2:
        return {}

    rets: Dict[str, List[float]] = {}
    prev_price = rows[0]["bid"]
    for r in rows[1:]:
        price = r["bid"]
        ts = datetime.strptime(r["time"], "%Y.%m.%d %H:%M:%S")
        key = ts.strftime("%Y-%m-%d %H" if interval.startswith("h") else "%Y-%m-%d")
        rets.setdefault(key, []).append(price - prev_price)
        prev_price = price

    vols = {}
    for key, vals in rets.items():
        if len(vals) > 1:
            vols[key] = float(statistics.pstdev(vals))
        else:
            vols[key] = 0.0
    return vols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tick_file")
    p.add_argument("--out", help="Output JSON file")
    p.add_argument("--vol-out", help="Write volatility JSON file")
    p.add_argument("--interval", choices=["daily", "hourly"], default="hourly")
    args = p.parse_args()

    rows = load_ticks(Path(args.tick_file))
    if args.vol_out:
        import json

        vols = compute_volatility(rows, interval=args.interval)
        with open(args.vol_out, "w") as f:
            json.dump(vols, f)

    stats = compute_metrics(rows)
    if args.out:
        import json

        with open(args.out, "w") as f:
            json.dump(stats, f)
    else:
        print(stats)


if __name__ == "__main__":
    main()
