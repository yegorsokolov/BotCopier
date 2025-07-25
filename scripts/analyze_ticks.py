#!/usr/bin/env python3
"""Compute basic metrics from exported tick history."""
import argparse
import csv
from pathlib import Path
from typing import Dict, List


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tick_file")
    p.add_argument("--out", help="Output JSON file")
    args = p.parse_args()

    rows = load_ticks(Path(args.tick_file))
    stats = compute_metrics(rows)
    if args.out:
        import json

        with open(args.out, "w") as f:
            json.dump(stats, f)
    else:
        print(stats)


if __name__ == "__main__":
    main()
