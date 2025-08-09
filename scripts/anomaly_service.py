#!/usr/bin/env python3
"""Simple HTTP service for autoencoder-based anomaly detection."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List
import numpy as np

model_data = {}

def load_model(path: Path) -> None:
    global model_data
    with open(path) as f:
        m = json.load(f)
    model_data = m.get("autoencoder", {})
    model_data["mean"] = np.array(model_data.get("mean", []), dtype=float)
    model_data["std"] = np.array(model_data.get("std", []), dtype=float)
    model_data["weights"] = [np.array(w, dtype=float) for w in model_data.get("weights", [])]
    model_data["bias"] = [np.array(b, dtype=float) for b in model_data.get("bias", [])]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - API requirement
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        try:
            vec = np.array(json.loads(data.decode()), dtype=float)
        except Exception:
            self.send_response(400)
            self.end_headers()
            return
        mean = model_data.get("mean")
        std = model_data.get("std")
        if mean is None or std is None or not model_data.get("weights"):
            self.send_response(500)
            self.end_headers()
            return
        x = (vec - mean) / std
        h = x
        weights: List[np.ndarray] = model_data["weights"]
        bias: List[np.ndarray] = model_data["bias"]
        for w, b in zip(weights[:-1], bias[:-1]):
            h = np.tanh(h @ w + b)
        recon = h @ weights[-1] + bias[-1]
        err = float(np.mean((x - recon) ** 2))
        thresh = float(model_data.get("threshold", 1.0))
        body = json.dumps({"error": err, "is_anomaly": err > thresh}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    p = argparse.ArgumentParser(description="Autoencoder anomaly detection service")
    p.add_argument("model", help="Path to model.json containing autoencoder")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    load_model(Path(args.model))
    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
