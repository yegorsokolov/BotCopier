"""Lightweight HTTP server exposing aggregated metrics."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Tuple

from .aggregator import get_aggregated_metrics


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # pragma: no cover - trivial
        if self.path.rstrip("/") == "/metrics":
            payload = json.dumps(get_aggregated_metrics()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args) -> None:  # pragma: no cover - silence
        return


def start_http_server(host: str = "127.0.0.1", port: int = 8000) -> Tuple[HTTPServer, threading.Thread]:
    """Start an HTTP server exposing metrics.

    Returns the server instance and the thread on which it is serving so it can
    later be shutdown cleanly.
    """

    server = HTTPServer((host, port), _MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread

