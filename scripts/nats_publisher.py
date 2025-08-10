#!/usr/bin/env python3
"""Publish trade or metric events to NATS JetStream."""
import argparse
import asyncio
import json
import nats
from proto import trade_event_pb2, metric_event_pb2

async def _publish(args) -> None:
    nc = await nats.connect(args.servers)
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
    await nc.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Publish a single event to NATS")
    p.add_argument("type", choices=["trade", "metric"], help="event type")
    p.add_argument("file", help="JSON file containing event fields")
    p.add_argument("--servers", default="nats://127.0.0.1:4222", help="NATS server URLs")
    p.add_argument("--schema-version", type=int, default=1, help="schema version byte")
    args = p.parse_args()
    asyncio.run(_publish(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
