#!/usr/bin/env python3
"""Bridge trade and metric messages to NATS JetStream.

This service listens for length-prefixed messages from ``Observer_TBot`` and
publishes the contained Protobuf payloads to NATS subjects. Messages begin with
one byte schema version followed by a byte indicating the type: ``0`` for
``TradeEvent`` and ``1`` for ``Metrics``.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Callable

import nats
from google.protobuf.message import DecodeError

from otel_logging import setup_logging
from proto import trade_event_pb2, metric_event_pb2

SCHEMA_VERSION = 1
TRADE_MSG = 0
METRIC_MSG = 1


async def _handle_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, js: nats.js.JetStreamContext, log: logging.Logger, tracer: Callable) -> None:
    try:
        while True:
            try:
                header = await reader.readexactly(4)
            except asyncio.IncompleteReadError:
                break
            length = int.from_bytes(header, "little")
            try:
                payload = await reader.readexactly(length)
            except asyncio.IncompleteReadError:
                break
            version = payload[0]
            msg_type = payload[1]
            data = payload[2:]
            if version != SCHEMA_VERSION:
                log.warning("schema version %d mismatch", version)
                continue
            if msg_type == TRADE_MSG:
                try:
                    trade_event_pb2.TradeEvent.FromString(data)
                except DecodeError:
                    continue
                await js.publish("trades", bytes([version]) + data)
                log.info("published trade %d bytes", len(data))
            elif msg_type == METRIC_MSG:
                try:
                    metric_event_pb2.MetricEvent.FromString(data)
                except DecodeError:
                    continue
                await js.publish("metrics", bytes([version]) + data)
                log.info("published metric %d bytes", len(data))
    finally:
        writer.close()
        await writer.wait_closed()


def main() -> int:
    p = argparse.ArgumentParser(description="Proxy observer messages to NATS")
    p.add_argument("--listen", default="127.0.0.1:6000", help="host:port to accept messages from Observer_TBot")
    p.add_argument("--servers", default="nats://127.0.0.1:4222", help="NATS server URLs")
    p.add_argument("--log-level", default="INFO", help="logging level")
    args = p.parse_args()

    tracer = setup_logging("nats_stream", getattr(logging, args.log_level.upper(), logging.INFO))
    log = logging.getLogger("nats_stream")

    async def _run() -> None:
        nc = await nats.connect(args.servers)
        js = nc.jetstream()
        host, port = args.listen.split(":")
        server = await asyncio.start_server(lambda r, w: _handle_conn(r, w, js, log, tracer), host, int(port))
        async with server:
            await server.serve_forever()

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
