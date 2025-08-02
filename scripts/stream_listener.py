#!/usr/bin/env python3
"""Simple socket listener that converts JSON messages to a CSV log."""

import argparse
import socket
from pathlib import Path
import csv
import json
import logging


EXPECTED_SCHEMA_VERSION = 1


FIELDS = [
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
    "remaining_lots",
]


def _write_lines(conn: socket.socket, out_file: Path) -> None:
    """Read newline-delimited JSON messages from ``conn`` and append rows."""

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        need_header = f.tell() == 0
        if need_header:
            writer.writerow(FIELDS)
        buffer = b""
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                schema = obj.get("schema_version")
                if schema != EXPECTED_SCHEMA_VERSION:
                    logging.warning(
                        "Schema version %s does not match expected %s",
                        schema,
                        EXPECTED_SCHEMA_VERSION,
                    )
                row = [str(obj.get(field, "")) for field in FIELDS]
                writer.writerow(row)
                f.flush()


def listen_once(host: str, port: int, out_file: Path) -> None:
    """Accept a single connection and process it until closed."""

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((host, port))
    srv.listen(1)
    try:
        conn, _ = srv.accept()
        with conn:
            _write_lines(conn, out_file)
    finally:
        srv.close()


def serve(host: str, port: int, out_file: Path) -> None:
    """Continually accept connections and append incoming lines."""

    while True:
        listen_once(host, port, out_file)


def main() -> None:
    p = argparse.ArgumentParser(description="Listen on socket and append to CSV")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--out", required=True, help="output CSV file")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)

    serve(args.host, args.port, Path(args.out))


if __name__ == "__main__":
    main()
