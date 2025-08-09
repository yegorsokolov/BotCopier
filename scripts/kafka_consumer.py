#!/usr/bin/env python3
"""Consume Avro encoded records from Kafka topics."""

import argparse
import json
from io import BytesIO
from pathlib import Path

from confluent_kafka import Consumer
from fastavro import parse_schema, schemaless_reader


def main() -> int:
    p = argparse.ArgumentParser(description="Consume Avro messages from Kafka")
    p.add_argument("topic", help="Kafka topic name")
    p.add_argument("schema", type=Path, help="Path to Avro schema (.avsc)")
    p.add_argument("--brokers", default="127.0.0.1:9092", help="Kafka bootstrap servers")
    p.add_argument("--group", default="bot", help="Consumer group id")
    args = p.parse_args()

    with args.schema.open() as f:
        schema = parse_schema(json.load(f))

    consumer = Consumer(
        {
            "bootstrap.servers": args.brokers,
            "group.id": args.group,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([args.topic])

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"error: {msg.error()}")
                continue
            buf = BytesIO(msg.value())
            record = schemaless_reader(buf, schema)
            print(json.dumps(record))
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
