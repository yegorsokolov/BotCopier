#!/usr/bin/env python3
"""Background service that writes JSON trade events from a TCP socket to CSV."""

import argparse
import socket
import json
import csv
from pathlib import Path

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
    """Read newline-delimited JSON from ``conn`` and append rows to ``out_file``."""

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
                row = [str(obj.get(field, "")) for field in FIELDS]
                writer.writerow(row)
                f.flush()


def listen_once(host: str, port: int, out_file: Path) -> None:
    """Accept a single connection and process messages until closed."""

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
    """Continually accept connections and append incoming log entries."""

    while True:
        listen_once(host, port, out_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write streamed log events to CSV")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--out", required=True, help="output CSV file")
    args = parser.parse_args()

    serve(args.host, args.port, Path(args.out))


if __name__ == "__main__":
    main()
