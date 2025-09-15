#!/usr/bin/env python3
"""Summarize trading and metric logs.

Reads ``logs/trades_raw.csv`` and ``logs/metrics.csv`` computing
win rate, Sharpe ratio, average hold time, slippage statistics and
model prediction accuracy.  A JSON summary is written and a one-line
CSV summary is appended to ``logs/summaries.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)


def _shadow_accuracy_vectorized(
    decisions: pd.DataFrame, trades: pd.DataFrame, n_jobs: int = 1
) -> float:
    """Compute shadow prediction accuracy without explicit Python loops."""
    if decisions.empty:
        return 0.0

    dec = decisions.sort_values("decision_id").merge(
        trades[["decision_id", "profit"]], on="decision_id", how="left"
    )
    if dec.empty:
        return 0.0

    rev = dec.iloc[::-1]
    groups = (
        (rev["action"] != "shadow")
        | (rev["executed_model_idx"] != rev["executed_model_idx"].shift())
    ).cumsum()
    dec["group"] = groups.iloc[::-1].to_numpy()

    exec_rows = dec[dec["action"].isin(["buy", "sell"])]
    if exec_rows.empty:
        return 0.0
    actual_map = {
        int(k): int(v)
        for k, v in (
            (exec_rows.set_index("group")["profit"].astype(float) > 0).astype(int)
        ).items()
    }

    shadow_rows = dec[dec["action"] == "shadow"]
    if shadow_rows.empty:
        return 0.0

    def chunk_acc(df_chunk: pd.DataFrame) -> tuple[int, int]:
        preds = (df_chunk["probability"].astype(float) > 0.5).astype(int)
        actuals = df_chunk["group"].map(actual_map)
        mask = actuals.notna()
        correct = int((preds[mask] == actuals[mask]).sum())
        total = int(mask.sum())
        return correct, total

    if n_jobs and n_jobs > 1:
        group_ids = shadow_rows["group"].unique()
        chunks = np.array_split(group_ids, n_jobs)
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(
                ex.map(
                    lambda g: chunk_acc(shadow_rows[shadow_rows["group"].isin(g)]),
                    chunks,
                )
            )
        correct = sum(r[0] for r in results)
        total = sum(r[1] for r in results)
    else:
        correct, total = chunk_acc(shadow_rows)

    return float(correct / total) if total else 0.0


def _compute_summary(
    trades: pd.DataFrame,
    metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    n_jobs: int = 1,
    benchmark: bool = False,
) -> dict:
    summary: dict[str, float] = {}

    if not trades.empty and "profit" in trades.columns:
        profits = trades["profit"].astype(float)
        summary["win_rate"] = float((profits > 0).mean())
        std = profits.std(ddof=0)
        summary["sharpe"] = (
            float(profits.mean() / std * (len(profits) ** 0.5)) if std > 0 else 0.0
        )
    else:
        summary["win_rate"] = 0.0
        summary["sharpe"] = 0.0

    if {"entry_time", "exit_time"}.issubset(trades.columns):
        entry = pd.to_datetime(trades["entry_time"])
        exit_ = pd.to_datetime(trades["exit_time"])
        hold = (exit_ - entry).dt.total_seconds().dropna()
        summary["avg_hold_time"] = float(hold.mean()) if not hold.empty else 0.0
    else:
        summary["avg_hold_time"] = 0.0

    if "slippage" in trades.columns:
        slip = trades["slippage"].astype(float)
        summary["slippage_mean"] = float(slip.mean())
        summary["slippage_std"] = float(slip.std(ddof=0))
    else:
        summary["slippage_mean"] = 0.0
        summary["slippage_std"] = 0.0

    if "decision_id" in trades.columns and {
        "decision_id",
        "prediction",
    }.issubset(metrics.columns):
        merged = trades[["decision_id", "profit"]].merge(
            metrics[["decision_id", "prediction"]], on="decision_id", how="inner"
        )
        if not merged.empty:
            merged["actual"] = (merged["profit"].astype(float) > 0).astype(int)
            merged["pred"] = (
                (merged["prediction"].astype(float) > 0.5).astype(int)
                if merged["prediction"].dtype != int
                else merged["prediction"].astype(int)
            )
            summary["prediction_accuracy"] = float(
                (merged["actual"] == merged["pred"]).mean()
            )
        else:
            summary["prediction_accuracy"] = 0.0
    else:
        summary["prediction_accuracy"] = 0.0

    if not trades.empty and not decisions.empty:
        if "decision_id" not in decisions.columns and "event_id" in decisions.columns:
            decisions = decisions.rename(columns={"event_id": "decision_id"})
        if {
            "decision_id",
            "action",
            "model_idx",
            "executed_model_idx",
            "probability",
        }.issubset(decisions.columns):
            exec_dec = decisions[decisions["action"].isin(["buy", "sell"])]
            merged = exec_dec.merge(
                trades[["decision_id", "profit"]], on="decision_id", how="left"
            )
            if not merged.empty:
                merged["actual"] = (merged["profit"].astype(float) > 0).astype(int)
                merged["pred"] = (
                    (merged["probability"].astype(float) > 0.5).astype(int)
                )
                summary["live_prediction_accuracy"] = float(
                    (merged["actual"] == merged["pred"]).mean()
                )
                if benchmark and n_jobs and n_jobs > 1:
                    t0 = time.perf_counter()
                    _shadow_accuracy_vectorized(decisions, trades, n_jobs=1)
                    t1 = time.perf_counter()
                    shadow_acc = _shadow_accuracy_vectorized(decisions, trades, n_jobs)
                    t2 = time.perf_counter()
                    if t2 - t1 > 0:
                        logger.info(
                            "shadow accuracy speedup %.2fx", (t1 - t0) / (t2 - t1)
                        )
                else:
                    shadow_acc = _shadow_accuracy_vectorized(decisions, trades, n_jobs)
                summary["shadow_prediction_accuracy"] = shadow_acc
            else:
                summary["live_prediction_accuracy"] = 0.0
                summary["shadow_prediction_accuracy"] = 0.0
        else:
            summary["live_prediction_accuracy"] = 0.0
            summary["shadow_prediction_accuracy"] = 0.0
    else:
        summary["live_prediction_accuracy"] = 0.0
        summary["shadow_prediction_accuracy"] = 0.0

    return summary


def _write_outputs(summary: dict, summary_file: Path, summaries_file: Path) -> None:
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)

    summaries_file.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now(timezone.utc).isoformat(), **summary}
    exists = summaries_file.exists()
    with summaries_file.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize trade logs")
    p.add_argument("--trades-file", default="logs/trades_raw.csv")
    p.add_argument("--metrics-file", default="logs/metrics.csv")
    p.add_argument("--summary-file", default="session_summary.json")
    p.add_argument("--summaries-file", default="logs/summaries.csv")
    p.add_argument("--decisions-file", default="logs/decisions.csv")
    p.add_argument("--n-jobs", type=int, default=1, help="parallel workers")
    p.add_argument("--benchmark", action="store_true", help="log speedup")
    args = p.parse_args(argv)

    trades = pd.read_csv(Path(args.trades_file))
    metrics = pd.read_csv(Path(args.metrics_file))
    decisions_path = Path(args.decisions_file)
    decisions = (
        pd.read_csv(decisions_path) if decisions_path.exists() else pd.DataFrame()
    )
    summary = _compute_summary(
        trades, metrics, decisions, n_jobs=args.n_jobs, benchmark=args.benchmark
    )
    _write_outputs(summary, Path(args.summary_file), Path(args.summaries_file))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
