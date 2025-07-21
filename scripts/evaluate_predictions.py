#!/usr/bin/env python3
"""Evaluate prediction accuracy from logs."""
import csv
import argparse
from pathlib import Path


def evaluate(log_file: Path):
    with open(log_file, newline='') as f:
        reader = csv.reader(f, delimiter=';')
        rows = list(reader)
    print(f"Loaded {len(rows)} rows from {log_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log_file')
    args = p.parse_args()
    evaluate(Path(args.log_file))

if __name__ == '__main__':
    main()
