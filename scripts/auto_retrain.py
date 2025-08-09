#!/usr/bin/env python3
"""Automatically retrain when live metrics degrade."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Optional

from scripts.train_target_clone import train as train_model
from scripts.backtest_strategy import run_backtest
from scripts.publish_model import publish

STATE_FILE = "last_event_id"


def _read_last_event_id(out_dir: Path) -> int:
    """Return the last processed event id."""
    try:
        return int((out_dir / STATE_FILE).read_text().strip())
    except Exception:
        return 0


def _write_last_event_id(out_dir: Path, event_id: int) -> None:
    (out_dir / STATE_FILE).write_text(str(int(event_id)))


def _load_latest_metrics(metrics_file: Path) -> Optional[Dict[str, float]]:
    if not metrics_file.exists():
        return None
    last: Optional[Dict[str, str]] = None
    with open(metrics_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            last = row
    if not last:
        return None
    try:
        return {
            "win_rate": float(last.get("win_rate", 0) or 0),
            "drawdown": float(last.get("drawdown", 0) or 0),
        }
    except Exception:
        return None


def retrain_if_needed(
    log_dir: Path,
    out_dir: Path,
    files_dir: Path,
    *,
    metrics_file: Optional[Path] = None,
    win_rate_threshold: float = 0.4,
    drawdown_threshold: float = 0.2,
    tick_file: Optional[Path] = None,
) -> bool:
    """Retrain and publish a model when win rate or drawdown worsens."""
    metrics_path = metrics_file or (log_dir / "metrics.csv")
    metrics = _load_latest_metrics(metrics_path)
    if not metrics:
        return False

    if metrics["win_rate"] >= win_rate_threshold and metrics["drawdown"] <= drawdown_threshold:
        return False

    # Load last processed event id for incremental training
    import scripts.train_target_clone as tc

    last_id = _read_last_event_id(out_dir)
    tc.START_EVENT_ID = last_id
    train_model(log_dir, out_dir, incremental=True)

    model_json = out_dir / "model.json"
    model_onnx = out_dir / "model.onnx"
    try:
        data = json.loads(model_json.read_text())
        _write_last_event_id(out_dir, int(data.get("last_event_id", last_id)))
    except Exception:
        pass

    # Backtest to ensure new model improves over previous metrics
    backtest_file = tick_file or (log_dir / "trades_raw.csv")
    try:
        result = run_backtest(model_json, backtest_file)
    except Exception:
        return False

    if result.get("win_rate", 0) <= metrics["win_rate"] or result.get("drawdown", 1) >= metrics["drawdown"]:
        return False

    publish(model_onnx if model_onnx.exists() else model_json, files_dir)
    tc.START_EVENT_ID = 0
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Retrain model when metrics degrade")
    p.add_argument("--log-dir", required=True, help="directory with observer logs")
    p.add_argument("--out-dir", required=True, help="output model directory")
    p.add_argument("--files-dir", required=True, help="MT4 Files directory")
    p.add_argument("--metrics-file", help="path to metrics.csv")
    p.add_argument("--tick-file", help="tick or trades file for backtesting")
    p.add_argument("--win-rate-threshold", type=float, default=0.4)
    p.add_argument("--drawdown-threshold", type=float, default=0.2)
    p.add_argument("--interval", type=float, help="seconds between checks (loop)" )
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    files_dir = Path(args.files_dir)
    metrics_path = Path(args.metrics_file) if args.metrics_file else None
    tick_path = Path(args.tick_file) if args.tick_file else None

    if args.interval:
        while True:
            retrain_if_needed(
                log_dir,
                out_dir,
                files_dir,
                metrics_file=metrics_path,
                win_rate_threshold=args.win_rate_threshold,
                drawdown_threshold=args.drawdown_threshold,
                tick_file=tick_path,
            )
            time.sleep(args.interval)
    else:
        retrain_if_needed(
            log_dir,
            out_dir,
            files_dir,
            metrics_file=metrics_path,
            win_rate_threshold=args.win_rate_threshold,
            drawdown_threshold=args.drawdown_threshold,
            tick_file=tick_path,
        )


if __name__ == "__main__":
    main()
