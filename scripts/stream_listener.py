#!/usr/bin/env python3
"""Simple socket listener that writes incoming lines to a CSV file."""

import argparse
import socket
from pathlib import Path


def _write_lines(conn: socket.socket, out_file: Path) -> None:
    """Read newline-delimited messages from ``conn`` and append to ``out_file``."""

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "a", newline="") as f:
        buffer = b""
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.decode("utf-8", errors="replace").strip()
                if line:
                    f.write(line + "\n")
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

    serve(args.host, args.port, Path(args.out))


if __name__ == "__main__":
    main()
