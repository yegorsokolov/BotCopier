import asyncio
import math
from pathlib import Path
import sys

import grpc

# Import generated proto modules
PROTO_DIR = Path(__file__).resolve().parents[1] / "proto"
sys.path.append(str(PROTO_DIR))
import predict_pb2  # type: ignore
import predict_pb2_grpc  # type: ignore

from botcopier.scripts import grpc_predict_server as gps


def test_grpc_round_trip():
    model = {"entry_coefficients": [0.5, -0.25], "entry_intercept": 0.1}
    gps.MODEL = model
    gps.COEFFS = [0.5, -0.25]
    gps.INTERCEPT = 0.1

    async def _run() -> None:
        server = await gps.create_server("127.0.0.1", 50070)
        async with grpc.aio.insecure_channel("127.0.0.1:50070") as channel:
            stub = predict_pb2_grpc.PredictServiceStub(channel)
            req = predict_pb2.PredictRequest(features=[1.0, 2.0])
            resp = await stub.Predict(req)
        await server.stop(None)
        expected = gps._predict_one([1.0, 2.0])
        assert math.isclose(resp.prediction, expected)

    asyncio.run(_run())
