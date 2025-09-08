#!/usr/bin/env python3
"""Periodically run ``label_uncertain.py`` to label low-confidence decisions.

This lightweight scheduler calls :mod:`scripts.label_uncertain` at a fixed
interval.  It can optionally start a tiny web UI that allows labeling via a
browser.  The web UI simply forwards submitted labels to ``label_uncertain.py``
so the implementation stays minimal and dependency free.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

SCRIPT = Path(__file__).resolve().parent / "label_uncertain.py"


def _run_labeler(input_file: Path, output_file: Path, label: int | None) -> None:
    cmd = [sys.executable, str(SCRIPT), str(input_file), str(output_file)]
    if label is not None:
        cmd += ["--label", str(label)]
    subprocess.run(cmd, check=True)


class _Handler(BaseHTTPRequestHandler):  # pragma: no cover - simple utility
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        body = (
            "<html><body><form method='post'>Label (0/1):"
            "<input name='label'/><input type='submit'/></form></body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length).decode()
        label = parse_qs(data).get("label", ["0"])[0]
        try:
            lbl = int(label)
        except ValueError:
            lbl = 0
        _run_labeler(self.server.input_file, self.server.output_file, lbl)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()


def main() -> None:
    p = argparse.ArgumentParser(description="Schedule label_uncertain runs")
    p.add_argument("--interval", type=float, default=3600.0, help="seconds between runs")
    p.add_argument("--input", type=Path, default=Path("uncertain_decisions.csv"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("uncertain_decisions_labeled.csv"),
        help="output CSV with labels",
    )
    p.add_argument("--label", type=int, choices=[0, 1], help="constant label to apply")
    p.add_argument("--once", action="store_true", help="run once then exit")
    p.add_argument("--host", default=None, help="optional host for web UI")
    p.add_argument("--port", type=int, default=8000, help="port for web UI")
    args = p.parse_args()

    if args.host:
        server = HTTPServer((args.host, args.port), _Handler)
        server.input_file = args.input  # type: ignore[attr-defined]
        server.output_file = args.output  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

    while True:
        _run_labeler(args.input, args.output, args.label)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
