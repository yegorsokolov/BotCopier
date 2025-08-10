#!/usr/bin/env python3
"""Arrow Flight server exposing trade and metric streams.

The server stores incoming record batches in memory and supports two
paths:

* ``trades``
* ``metrics``
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import pyarrow as pa
import pyarrow.flight as flight

from schemas import TRADE_SCHEMA, METRIC_SCHEMA


class FlightServer(flight.FlightServerBase):
    """Simple in-memory Flight server for trades and metrics."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8815) -> None:
        self._host = host
        location = f"grpc://{host}:{port}"
        super().__init__(location)
        self._batches: Dict[str, List[pa.RecordBatch]] = {
            "trades": [],
            "metrics": [],
        }

    # ------------------------------------------------------------------
    def get_flight_info(
        self, context: flight.ServerCallContext, descriptor: flight.FlightDescriptor
    ) -> flight.FlightInfo:
        path = descriptor.path[0].decode()
        if path not in ("trades", "metrics"):
            raise KeyError(f"unknown path: {path}")
        schema = TRADE_SCHEMA if path == "trades" else METRIC_SCHEMA
        ticket = flight.Ticket(path.encode())
        endpoint = flight.FlightEndpoint(ticket, [flight.Location.for_grpc_tcp(self._host, self.port)])
        return flight.FlightInfo(schema, descriptor, [endpoint], -1, -1)

    # ------------------------------------------------------------------
    def do_put(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.RecordBatchStream,
        writer: flight.ServerStreamWriter,
    ) -> None:
        path = descriptor.path[0].decode()
        if path not in ("trades", "metrics"):
            raise KeyError(f"unknown path: {path}")
        batches = self._batches.setdefault(path, [])
        for chunk in reader:
            batches.append(chunk.data)

    # ------------------------------------------------------------------
    def do_get(
        self, context: flight.ServerCallContext, ticket: flight.Ticket
    ) -> flight.RecordBatchStream:
        path = ticket.ticket.decode()
        if path not in ("trades", "metrics"):
            raise KeyError(f"unknown path: {path}")
        schema = TRADE_SCHEMA if path == "trades" else METRIC_SCHEMA
        table = pa.Table.from_batches(self._batches.get(path, []), schema=schema)
        return flight.RecordBatchStream(table)


def main() -> None:
    p = argparse.ArgumentParser(description="Arrow Flight log server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8815)
    args = p.parse_args()
    server = FlightServer(args.host, args.port)
    server.serve()


if __name__ == "__main__":
    main()
