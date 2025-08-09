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
from sklearn.ensemble import IsolationForest

from otel_logging import setup_logging


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
                if check_anomaly(df, args.metric, args.method, args.threshold):
                    body = f"Anomaly detected for {args.metric}: {df[args.metric].iloc[-1]}"
                    log.warning(body)
                    if args.email:
                        _send_email(args.smtp_server, args.smtp_port, args.email, body)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
