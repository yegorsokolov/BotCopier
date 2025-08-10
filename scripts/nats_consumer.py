#!/usr/bin/env python3
"""Consume NATS events and write them to CSV and/or SQLite."""
from __future__ import annotations
import argparse
import asyncio
import csv
import logging
import sqlite3
from pathlib import Path

import nats
from proto import trade_event_pb2, metric_event_pb2

logger = logging.getLogger("nats_consumer")


def _setup_csv(path: Path, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = path.open("a", newline="")
    writer = csv.DictWriter(fp, fieldnames)
    if fp.tell() == 0:
        writer.writeheader()
    return fp, writer


def _setup_sqlite(path: Path, table: str, fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cols = ",".join(f"{f} TEXT" for f in fieldnames)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
    placeholders = ",".join(["?"] * len(fieldnames))
    insert_sql = f"INSERT INTO {table} ({','.join(fieldnames)}) VALUES ({placeholders})"
    return conn, insert_sql


def _row_from_msg(msg):
    return {f.name: getattr(msg, f.name) for f in msg.DESCRIPTOR.fields}


async def _consume(js, subject, proto_cls, schema_version, csv_writer=None, db=None, insert_sql=None):
    sub = await js.subscribe(subject, durable=f"{subject}_consumer")
    async for m in sub.messages:
        version = m.data[0]
        if version != schema_version:
            logger.warning("schema version mismatch: %s", version)
            await m.ack()
            continue
        obj = proto_cls.FromString(m.data[1:])
        row = _row_from_msg(obj)
        if csv_writer:
            csv_writer.writerow(row)
        if db:
            db.execute(insert_sql, [row[k] for k in row])
            db.commit()
        await m.ack()


async def _run(args):
    nc = await nats.connect(args.servers)
    js = nc.jetstream()
    tasks = []
    trade_fields = [f.name for f in trade_event_pb2.TradeEvent.DESCRIPTOR.fields]
    metric_fields = [f.name for f in metric_event_pb2.MetricEvent.DESCRIPTOR.fields]

    trade_csv_fp = trade_csv_writer = None
    metric_csv_fp = metric_csv_writer = None
    trade_db = trade_insert = None
    metric_db = metric_insert = None

    if args.trades_csv:
        trade_csv_fp, trade_csv_writer = _setup_csv(Path(args.trades_csv), trade_fields)
    if args.metrics_csv:
        metric_csv_fp, metric_csv_writer = _setup_csv(Path(args.metrics_csv), metric_fields)
    if args.trades_sqlite:
        trade_db, trade_insert = _setup_sqlite(Path(args.trades_sqlite), "trades", trade_fields)
    if args.metrics_sqlite:
        metric_db, metric_insert = _setup_sqlite(Path(args.metrics_sqlite), "metrics", metric_fields)

    if trade_csv_writer or trade_db:
        tasks.append(asyncio.create_task(
            _consume(js, "trades", trade_event_pb2.TradeEvent, args.schema_version,
                     trade_csv_writer, trade_db, trade_insert)))
    if metric_csv_writer or metric_db:
        tasks.append(asyncio.create_task(
            _consume(js, "metrics", metric_event_pb2.MetricEvent, args.schema_version,
                     metric_csv_writer, metric_db, metric_insert)))

    if tasks:
        await asyncio.gather(*tasks)

    if trade_csv_fp:
        trade_csv_fp.close()
    if metric_csv_fp:
        metric_csv_fp.close()
    if trade_db:
        trade_db.close()
    if metric_db:
        metric_db.close()
    await nc.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Consume NATS events for training")
    p.add_argument("--servers", default="nats://127.0.0.1:4222")
    p.add_argument("--schema-version", type=int, default=1)
    p.add_argument("--trades-csv")
    p.add_argument("--metrics-csv")
    p.add_argument("--trades-sqlite")
    p.add_argument("--metrics-sqlite")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
