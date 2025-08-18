#!/usr/bin/env python3
"""Label uncertain model decisions.

Reads ``uncertain_decisions.csv`` and writes ``uncertain_decisions_labeled.csv``
with an additional ``label`` column. Labels may be provided interactively or
set for all rows via ``--label`` to support simple heuristic labeling.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Label uncertain decisions")
    parser.add_argument(
        "input",
        nargs="?",
        default="uncertain_decisions.csv",
        help="CSV file with uncertain decisions",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="uncertain_decisions_labeled.csv",
        help="output CSV file with labels",
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        help="assign this label to all rows instead of prompting",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows: list[dict[str, str]] = []
    with in_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames or []
        for row in reader:
            if args.label is None:
                info = {
                    "action": row.get("action", ""),
                    "prob": row.get("probability", ""),
                    "thr": row.get("threshold", ""),
                }
                print(f"Action: {info['action']} prob={info['prob']} thr={info['thr']}")
                print(f"Features: {row.get('features', '')}")
                lbl = input("Label (0/1): ").strip() or "0"
            else:
                lbl = str(args.label)
            row["label"] = lbl
            rows.append(row)
    if "label" not in fieldnames:
        fieldnames.append("label")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
