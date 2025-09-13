#!/usr/bin/env python3
"""Example gRPC client for the Predict service."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import grpc

# Ensure generated modules are importable
PROTO_DIR = Path(__file__).resolve().parent.parent / "proto"
sys.path.append(str(PROTO_DIR))
import predict_pb2  # type: ignore
import predict_pb2_grpc  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict via gRPC")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("features", help="Comma separated feature values")
    args = parser.parse_args()

    feats = [float(x) for x in args.features.split(",") if x]
    with grpc.insecure_channel(f"{args.host}:{args.port}") as channel:
        stub = predict_pb2_grpc.PredictServiceStub(channel)
        resp = stub.Predict(predict_pb2.PredictRequest(features=feats))
    print(resp.prediction)


if __name__ == "__main__":
    main()
