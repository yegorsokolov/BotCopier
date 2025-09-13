#!/usr/bin/env python3
"""gRPC service exposing a simple model prediction RPC."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import List

import grpc.aio as grpc

# Make generated proto modules importable
PROTO_DIR = Path(__file__).resolve().parent.parent.parent / "proto"
import sys
sys.path.append(str(PROTO_DIR))
import predict_pb2  # type: ignore
import predict_pb2_grpc  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "model.json"


def _load_model(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {"feature_names": [], "entry_coefficients": [], "entry_intercept": 0.0}


MODEL = _load_model(DEFAULT_MODEL_PATH)
COEFFS: List[float] = [float(x) for x in MODEL.get("entry_coefficients", [])]
INTERCEPT: float = float(MODEL.get("entry_intercept", 0.0))


def _predict_one(features: List[float]) -> float:
    if len(features) != len(COEFFS):
        raise ValueError("feature length mismatch")
    score = sum(c * f for c, f in zip(COEFFS, features)) + INTERCEPT
    return 1.0 / (1.0 + math.exp(-score))


class _PredictService(predict_pb2_grpc.PredictServiceServicer):
    async def Predict(self, request, context):  # noqa: N802 gRPC naming
        try:
            pred = _predict_one(list(request.features))
            return predict_pb2.PredictResponse(prediction=pred)
        except ValueError as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))


async def create_server(host: str, port: int) -> grpc.Server:
    server = grpc.server()
    predict_pb2_grpc.add_PredictServiceServicer_to_server(_PredictService(), server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    return server


async def serve(host: str, port: int) -> None:
    server = await create_server(host, port)
    await server.wait_for_termination()


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="gRPC predict service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    global MODEL, COEFFS, INTERCEPT
    MODEL = _load_model(args.model)
    COEFFS = [float(x) for x in MODEL.get("entry_coefficients", [])]
    INTERCEPT = float(MODEL.get("entry_intercept", 0.0))

    await serve(args.host, args.port)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
