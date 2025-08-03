#!/usr/bin/env python3
"""Listen for JSON messages from the observer EA and append them to CSV logs.

The script expects newline-delimited JSON objects on ``stdin``. Each message
must include a ``schema_version`` field matching the expected version and a
``type`` field indicating either ``"event"`` or ``"metric"``. All remaining
fields are written directly to the appropriate CSV file under ``logs/``.

Set the ``SCHEMA_VERSION`` environment variable to override the default
version.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zlib
from pathlib import Path

# Expected schema version for incoming messages. Can be overridden via env var.
EXPECTED_SCHEMA_VERSION = os.environ.get("SCHEMA_VERSION", "1.0")

# Mapping from message type to output CSV file.
LOG_FILES = {
    "event": Path("logs/trades_raw.csv"),
    "metric": Path("logs/metrics.csv"),
}


def append_csv(path: Path, record: dict) -> None:
    """Append ``record`` to ``path`` writing a header row when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def process_message(message: dict) -> None:
    """Validate and route a message to the appropriate CSV file."""
    schema = message.get("schema_version")
    if schema != EXPECTED_SCHEMA_VERSION:
        print(
            f"Unsupported schema_version: {schema}; expected {EXPECTED_SCHEMA_VERSION}",
            file=sys.stderr,
        )
        return

    msg_type = message.get("type")
    if msg_type not in LOG_FILES:
        print(f"Unknown message type: {msg_type}", file=sys.stderr)
        return

    record = {k: v for k, v in message.items() if k not in {"schema_version", "type"}}
    append_csv(LOG_FILES[msg_type], record)


def main() -> int:
    p = argparse.ArgumentParser(description="Process observer stream logs")
    p.add_argument(
        "--binary",
        action="store_true",
        help="expect length-prefixed gzipped JSON records",
    )
    args = p.parse_args()

    if args.binary:
        buf = sys.stdin.buffer
        while True:
            header = buf.read(4)
            if len(header) < 4:
                break
            length = int.from_bytes(header, "little")
            payload = buf.read(length)
            if len(payload) < length:
                break
            try:
                line = zlib.decompress(payload).decode("utf-8")
            except Exception as exc:
                print(f"Invalid compressed payload: {exc}", file=sys.stderr)
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON: {exc}", file=sys.stderr)
                continue
            process_message(message)
    else:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON: {exc}", file=sys.stderr)
                continue
            process_message(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
