#!/usr/bin/env python3
"""Publish trade or metric events to NATS JetStream."""
import argparse
import asyncio
import json

import nats

from proto import metric_event_pb2, trade_event_pb2


async def async_main(argv: list[str] | None = None) -> None:
    """Publish a single event to NATS using ``asyncio``."""
    p = argparse.ArgumentParser(description="Publish a single event to NATS")
    p.add_argument("type", choices=["trade", "metric"], help="event type")
    p.add_argument("file", help="JSON file containing event fields")
    p.add_argument(
        "--servers", default="nats://127.0.0.1:4222", help="NATS server URLs"
    )
    p.add_argument(
        "--schema-version", type=int, default=1, help="schema version byte"
    )
    args = p.parse_args(argv)

    nc = await nats.connect(args.servers)
    try:
        js = nc.jetstream()
        with open(args.file) as f:
            data = json.load(f)
        if args.type == "trade":
            msg = trade_event_pb2.TradeEvent(**data)
            subject = "trades"
        else:
            msg = metric_event_pb2.MetricEvent(**data)
            subject = "metrics"
        payload = bytes([args.schema_version]) + msg.SerializeToString()
        await js.publish(subject, payload)
    finally:
        await nc.close()


def main(argv: list[str] | None = None) -> int:
    asyncio.run(async_main(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
