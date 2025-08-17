#!/usr/bin/env python3
"""Arrow Flight server exposing trade and metric streams.

Incoming record batches are retained in memory for clients and mirrored
to both SQLite databases and Parquet datasets. Two logical paths are
served:

* ``trades``
* ``metrics``
"""

from __future__ import annotations

import argparse
from typing import Dict, List
import logging
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.flight as flight
import pyarrow.parquet as pq

from schemas import TRADE_SCHEMA, METRIC_SCHEMA

try:  # prefer systemd journal if available
    from systemd.journal import JournalHandler
    logging.basicConfig(handlers=[JournalHandler()], level=logging.INFO)
except Exception:  # pragma: no cover - fallback to stderr
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class FlightServer(flight.FlightServerBase):
    """Arrow Flight server persisting streams to disk.

    Incoming record batches are kept in memory for clients to retrieve
    and also appended to Parquet and SQLite storage.  A short summary of
    each batch is mirrored to the systemd journal for observability.
    """

    def __init__(
        self, host: str = "0.0.0.0", port: int = 8815, data_dir: str = "flight_logs"
    ) -> None:
        self._host = host
        location = f"grpc://{host}:{port}"
        super().__init__(location)
        self._batches: Dict[str, List[pa.RecordBatch]] = {
            "trades": [],
            "metrics": [],
        }
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sqlite: Dict[str, sqlite3.Connection] = {}
        self._sqlite_sql: Dict[str, str] = {}
        for name, schema in (("trades", TRADE_SCHEMA), ("metrics", METRIC_SCHEMA)):
            conn = sqlite3.connect(self._dir / f"{name}.db", check_same_thread=False)
            cols = [f.name for f in schema]
            col_defs = ",".join(f"{c} TEXT" for c in cols)
            conn.execute(f"CREATE TABLE IF NOT EXISTS {name} ({col_defs})")
            placeholders = ",".join(["?"] * len(cols))
            self._sqlite_sql[name] = f"INSERT INTO {name} ({','.join(cols)}) VALUES ({placeholders})"
            self._sqlite[name] = conn

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
        schema = TRADE_SCHEMA if path == "trades" else METRIC_SCHEMA
        for chunk in reader:
            batch = chunk.data
            batches.append(batch)
            table = pa.Table.from_batches([batch], schema=schema)
            # Parquet persistence
            pq.write_to_dataset(
                table, root_path=str(self._dir / path), basename_template="part-{i}.parquet"
            )
            # SQLite persistence without pandas
            rows = table.to_pylist()
            conn = self._sqlite[path]
            conn.executemany(self._sqlite_sql[path], [list(r.values()) for r in rows])
            conn.commit()
            logger.info("stored %d %s rows", batch.num_rows, path)

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

    # ------------------------------------------------------------------
    def shutdown(self) -> None:  # pragma: no cover - simple resource cleanup
        for conn in self._sqlite.values():
            try:
                conn.close()
            except Exception:
                pass
        super().shutdown()


def main() -> None:
    p = argparse.ArgumentParser(description="Arrow Flight log server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8815)
    p.add_argument("--data-dir", default="flight_logs", help="storage directory")
    args = p.parse_args()
    server = FlightServer(args.host, args.port, args.data_dir)
    server.serve()


if __name__ == "__main__":
    main()
