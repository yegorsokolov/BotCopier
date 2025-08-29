#!/usr/bin/env python3
"""gRPC service that logs trade and metric events to CSV files."""

from __future__ import annotations

import argparse
import csv
from concurrent import futures
from pathlib import Path
import sys

import grpc
from google.protobuf import empty_pb2

sys.path.append(str(Path(__file__).resolve().parent.parent / "proto"))
import log_service_pb2_grpc  # type: ignore
import metric_event_pb2  # type: ignore
import trade_event_pb2  # type: ignore

TRADE_FIELDS = [
    "event_id",
    "event_time",
    "broker_time",
    "local_time",
    "action",
    "ticket",
    "magic",
    "source",
    "symbol",
    "order_type",
    "lots",
    "price",
    "sl",
    "tp",
    "profit",
    "profit_after_trade",
    "spread",
    "comment",
    "remaining_lots",
    "slippage",
    "volume",
    "open_time",
    "book_bid_vol",
    "book_ask_vol",
    "book_imbalance",
    "sl_hit_dist",
    "tp_hit_dist",
    "decision_id",
]

METRIC_FIELDS = [
    "time",
    "magic",
    "win_rate",
    "avg_profit",
    "trade_count",
    "drawdown",
    "sharpe",
    "file_write_errors",
    "socket_errors",
    "book_refresh_seconds",
]


class _LogService(log_service_pb2_grpc.LogServiceServicer):
    def __init__(self, trade_out: Path, metrics_out: Path) -> None:
        self.trade_out = Path(trade_out)
        self.metrics_out = Path(metrics_out)
        self.trade_out.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, out: Path, fields: list[str], obj) -> None:
        first = not out.exists()
        with out.open("a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            if first:
                writer.writerow(fields)
            writer.writerow([getattr(obj, f) for f in fields])

    def LogTrade(self, request, context):  # noqa: N802 gRPC naming
        self._append(self.trade_out, TRADE_FIELDS, request)
        return empty_pb2.Empty()

    def LogMetrics(self, request, context):  # noqa: N802 gRPC naming
        self._append(self.metrics_out, METRIC_FIELDS, request)
        return empty_pb2.Empty()


def create_server(host: str, port: int, trade_out: Path, metrics_out: Path) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    log_service_pb2_grpc.add_LogServiceServicer_to_server(
        _LogService(trade_out, metrics_out),
        server,
    )
    server.add_insecure_port(f"{host}:{port}")
    return server


def serve(host: str, port: int, trade_out: Path, metrics_out: Path) -> None:
    server = create_server(host, port, trade_out, metrics_out)
    server.start()
    server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="gRPC log service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--trade-out", default="trades.csv", help="trade CSV output")
    parser.add_argument("--metrics-out", default="metrics.csv", help="metrics CSV output")
    args = parser.parse_args()

    serve(args.host, args.port, Path(args.trade_out), Path(args.metrics_out))


if __name__ == "__main__":
    main()
