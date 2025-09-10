#!/usr/bin/env python3
"""Monitor metrics for anomalies using statistical tests."""

from __future__ import annotations

import argparse
import logging
import smtplib
import sqlite3
import time
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import ruptures as rpt
from sklearn.ensemble import IsolationForest

try:  # pragma: no cover - fallback for package import
    from otel_logging import setup_logging
except ImportError:  # pragma: no cover
    from scripts.otel_logging import setup_logging
from opentelemetry import trace
from opentelemetry.trace import format_trace_id


def _send_email(server: str, port: int, to_addr: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = "Anomaly detected"
    msg["From"] = to_addr
    msg["To"] = to_addr
    msg.set_content(body)
    with smtplib.SMTP(server, port) as s:
        s.send_message(msg)


def _ewma_anomaly(series: pd.Series, threshold: float) -> bool:
    mean = series.ewm(alpha=0.3).mean()
    std = series.ewm(alpha=0.3).std().fillna(0)
    last = series.iloc[-1]
    return abs(last - mean.iloc[-1]) > threshold * (std.iloc[-1] or 1.0)


def _iforest_anomaly(series: pd.Series, threshold: float) -> bool:
    model = IsolationForest(contamination=threshold)
    preds = model.fit_predict(series.to_frame())
    return preds[-1] == -1


def _ruptures_change_point(
    series: pd.Series, penalty: float, window: int = 5
) -> bool:
    if len(series) < window * 2:
        return False
    algo = rpt.Pelt(model="rbf").fit(series.values)
    result = algo.predict(pen=penalty)
    if len(result) >= 2 and len(series) - result[-2] <= window:
        return True
    return False


def detect_change_points(
    df: pd.DataFrame, metrics: list[str], penalty: float, log: logging.Logger | None = None
) -> list[str]:
    hits: list[str] = []
    for metric in metrics:
        if metric not in df:
            continue
        series = df[metric].astype(float)
        if _ruptures_change_point(series, penalty):
            hits.append(metric)
            if log is not None:
                span = trace.get_current_span()
                trace_id = format_trace_id(span.get_span_context().trace_id)
                log.warning(
                    f"Change point detected for {metric} trace_id={trace_id}: {series.iloc[-1]}"
                )
    return hits


def check_anomaly(df: pd.DataFrame, metric: str, method: str, threshold: float) -> bool:
    series = df[metric].astype(float)
    if method == "ewma":
        return _ewma_anomaly(series, threshold)
    else:
        return _iforest_anomaly(series, threshold)


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor metrics for anomalies")
    p.add_argument("--db", required=True, help="SQLite DB produced by metrics_collector")
    p.add_argument("--metric", default="win_rate", help="metric column to monitor")
    p.add_argument(
        "--method",
        choices=["ewma", "isolation_forest"],
        default="ewma",
        help="anomaly detection method",
    )
    p.add_argument(
        "--threshold", type=float, default=3.0, help="EWMA stddev multiplier or contamination"
    )
    p.add_argument("--interval", type=int, default=60, help="polling interval in seconds")
    p.add_argument("--email", help="address to send alert emails to")
    p.add_argument("--smtp-server", default="localhost")
    p.add_argument("--smtp-port", type=int, default=25)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    tracer = setup_logging("anomaly_monitor", getattr(logging, args.log_level.upper(), logging.INFO))
    log = logging.getLogger("anomaly_monitor")

    last_row = 0
    db_path = Path(args.db)

    while True:
        with tracer.start_as_current_span("check_metrics"):
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(
                f"SELECT rowid, * FROM metrics WHERE rowid > {last_row}", conn
            )
            conn.close()
            if not df.empty:
                last_row = int(df["rowid"].max())
                hits = detect_change_points(
                    df, ["win_rate", "drawdown"], args.threshold, log
                )
                for metric in hits:
                    span = trace.get_current_span()
                    trace_id = format_trace_id(span.get_span_context().trace_id)
                    if args.email:
                        body = (
                            f"Change point detected for {metric} trace_id={trace_id}: "
                            f"{df[metric].iloc[-1]}"
                        )
                        _send_email(args.smtp_server, args.smtp_port, args.email, body)
                if check_anomaly(df, args.metric, args.method, args.threshold):
                    body = f"Anomaly detected for {args.metric}: {df[args.metric].iloc[-1]}"
                    log.warning(body)
                    if args.email:
                        _send_email(args.smtp_server, args.smtp_port, args.email, body)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
