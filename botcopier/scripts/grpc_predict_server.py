#!/usr/bin/env python3
"""gRPC service exposing a simple model prediction RPC."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import signal
from contextlib import suppress
from pathlib import Path
from typing import List

import grpc.aio as grpc

# Make generated proto modules importable
PROTO_DIR = Path(__file__).resolve().parent.parent.parent / "proto"
import sys
sys.path.append(str(PROTO_DIR))
import predict_pb2  # type: ignore
import predict_pb2_grpc  # type: ignore

from botcopier.exceptions import ModelError, ServiceError

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "model.json"

MODEL: dict = {"entry_coefficients": [], "entry_intercept": 0.0}
COEFFS: List[float] = []
INTERCEPT: float = 0.0

logger = logging.getLogger(__name__)


def _load_model(path: Path) -> dict:
    """Load a JSON model specification from ``path``."""

    try:
        with path.open("r", encoding="utf-8") as fh:
            model = json.load(fh)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise ModelError(f"Model file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ModelError(f"Model file is not valid JSON: {path}") from exc
    except OSError as exc:  # pragma: no cover - defensive guard
        raise ModelError(f"Unable to read model file: {path}") from exc

    if not isinstance(model, dict):
        raise ModelError(f"Model payload must be a JSON object: {path}")
    return model


def _reload_model(path: Path) -> None:
    """Refresh global model parameters from ``path``."""

    global MODEL, COEFFS, INTERCEPT
    model = _load_model(path)
    MODEL = model
    COEFFS = [float(x) for x in model.get("entry_coefficients", [])]
    INTERCEPT = float(model.get("entry_intercept", 0.0))
    logger.info(
        "model_loaded",
        extra={
            "context": {
                "coefficients": len(COEFFS),
                "intercept": INTERCEPT,
                "path": str(path),
            }
        },
    )


def _predict_one(features: List[float]) -> float:
    """Compute the logistic score for ``features``."""

    if len(features) != len(COEFFS):
        raise ServiceError("feature length mismatch")
    score = sum(c * f for c, f in zip(COEFFS, features)) + INTERCEPT
    return 1.0 / (1.0 + math.exp(-score))


def _build_server(host: str, port: int) -> grpc.Server:
    server = grpc.server()
    predict_pb2_grpc.add_PredictServiceServicer_to_server(_PredictService(), server)
    server.add_insecure_port(f"{host}:{port}")
    return server


class _ServerManager:
    """Async context manager ensuring the gRPC server is stopped."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self.server: grpc.Server | None = None
        self.stopped = False

    async def __aenter__(self) -> grpc.Server:
        server = _build_server(self._host, self._port)
        await server.start()
        logger.info(
            "server_started",
            extra={"context": {"host": self._host, "port": self._port}},
        )
        self.server = server
        return server

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.server and not self.stopped:
            await self.server.stop(None)
        logger.info(
            "server_stopped",
            extra={"context": {"host": self._host, "port": self._port}},
        )


class _PredictService(predict_pb2_grpc.PredictServiceServicer):
    async def Predict(self, request, context):  # noqa: N802 gRPC naming
        try:
            pred = _predict_one(list(request.features))
            return predict_pb2.PredictResponse(prediction=pred)
        except ServiceError as exc:
            logger.exception(
                "prediction_failed",
                extra={
                    "context": {
                        "feature_count": len(request.features),
                        "error": str(exc),
                    }
                },
            )
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception(
                "prediction_internal_error",
                extra={
                    "context": {
                        "feature_count": len(request.features),
                        "error": str(exc),
                    }
                },
            )
            await context.abort(grpc.StatusCode.INTERNAL, "internal error")


async def create_server(host: str, port: int) -> grpc.Server:
    server = _build_server(host, port)
    await server.start()
    logger.info(
        "server_started",
        extra={"context": {"host": host, "port": port}},
    )
    return server


async def serve(
    host: str,
    port: int,
    *,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    stop_event = shutdown_event or asyncio.Event()
    registered: list[signal.Signals] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
            registered.append(sig)
        except NotImplementedError:  # pragma: no cover - platform specific
            continue

    manager = _ServerManager(host, port)
    try:
        async with manager as server:
            wait_task = asyncio.create_task(server.wait_for_termination())
            stop_task = asyncio.create_task(stop_event.wait())
            try:
                done, _ = await asyncio.wait(
                    {wait_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if stop_task in done and not wait_task.done():
                    logger.info(
                        "shutdown_requested",
                        extra={"context": {"host": host, "port": port}},
                    )
                    await server.stop(None)
                    manager.stopped = True
                await wait_task
                manager.stopped = True
            finally:
                for task in (wait_task, stop_task):
                    if not task.done():
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
    finally:
        for sig in registered:
            loop.remove_signal_handler(sig)


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="gRPC predict service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    try:
        _reload_model(args.model)
    except ModelError:
        logger.exception(
            "model_load_failed", extra={"context": {"path": str(args.model)}}
        )
        raise SystemExit(1) from None

    await serve(args.host, args.port)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
