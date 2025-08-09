#!/usr/bin/env python3
"""Produce Avro encoded records to Kafka topics."""

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

from confluent_kafka import Producer
from fastavro import parse_schema, schemaless_writer


def main() -> int:
    p = argparse.ArgumentParser(description="Produce Avro messages to Kafka")
    p.add_argument("topic", help="Kafka topic name")
    p.add_argument("schema", type=Path, help="Path to Avro schema (.avsc)")
    p.add_argument("input", nargs="?", type=Path, help="JSON file with one record per line")
    p.add_argument("--brokers", default="127.0.0.1:9092", help="Kafka bootstrap servers")
    args = p.parse_args()

    with args.schema.open() as f:
        schema = parse_schema(json.load(f))

    producer = Producer({"bootstrap.servers": args.brokers})

    def _send(record: dict) -> None:
        buf = BytesIO()
        schemaless_writer(buf, schema, record)
        producer.produce(args.topic, buf.getvalue())

    fh = args.input.open() if args.input else sys.stdin
    try:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            _send(record)
    finally:
        if fh is not sys.stdin:
            fh.close()
    producer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
