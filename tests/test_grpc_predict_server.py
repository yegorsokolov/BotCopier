import asyncio
import logging
import math
import socket
from pathlib import Path
import sys

import pytest

grpc_mod = pytest.importorskip("grpc")
from grpc import aio as grpc_aio

# Import generated proto modules
PROTO_DIR = Path(__file__).resolve().parents[1] / "proto"
sys.path.append(str(PROTO_DIR))
import predict_pb2  # type: ignore
import predict_pb2_grpc  # type: ignore

from botcopier.scripts import grpc_predict_server as gps


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_grpc_round_trip():
    model = {
        "entry_coefficients": [0.5, -0.25],
        "entry_intercept": 0.1,
        "threshold": 0.0,
    }
    gps._configure_runtime(model)

    async def _run() -> None:
        port = _free_port()
        server = await gps.create_server("127.0.0.1", port)
        async with grpc_aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = predict_pb2_grpc.PredictServiceStub(channel)
            req = predict_pb2.PredictRequest(features=[1.0, 2.0])
            resp = await stub.Predict(req)
        await server.stop(None)
        expected = gps._predict_one([1.0, 2.0])
        assert math.isclose(resp.prediction, expected)

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_grpc_invalid_features_logged(caplog):
    gps._configure_runtime(
        {"entry_coefficients": [0.5, -0.25], "entry_intercept": 0.0, "threshold": 0.0}
    )
    caplog.set_level(logging.ERROR)

    port = _free_port()
    server = await gps.create_server("127.0.0.1", port)
    try:
        async with grpc_aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = predict_pb2_grpc.PredictServiceStub(channel)
            with pytest.raises(grpc_aio.AioRpcError) as excinfo:
                await stub.Predict(predict_pb2.PredictRequest(features=[1.0]))
        assert excinfo.value.code() == grpc_mod.StatusCode.INVALID_ARGUMENT
    finally:
        await server.stop(None)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "prediction_failed" in messages


@pytest.mark.asyncio
async def test_serve_shutdown_event_triggers_stop(monkeypatch):
    shutdown_event = asyncio.Event()

    class DummyServer:
        def __init__(self) -> None:
            self._terminated = asyncio.Event()
            self.stop_called = False
            self.wait_called = False

        def add_insecure_port(self, address: str) -> None:  # pragma: no cover - helper
            self.address = address

        async def start(self) -> None:
            return None

        async def wait_for_termination(self) -> None:
            self.wait_called = True
            await self._terminated.wait()

        async def stop(self, grace) -> None:
            self.stop_called = True
            self._terminated.set()

    dummy = DummyServer()
    monkeypatch.setattr(gps, "_build_server", lambda host, port: dummy)

    serve_task = asyncio.create_task(
        gps.serve("127.0.0.1", 50072, shutdown_event=shutdown_event)
    )
    await asyncio.sleep(0.05)
    shutdown_event.set()
    await asyncio.wait_for(serve_task, timeout=1.0)

    assert dummy.stop_called
    assert dummy.wait_called
