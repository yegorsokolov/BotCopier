#!/usr/bin/env python3
"""Aggregate decision logs to evaluate meta strategy performance.

This utility reads ``decisions.csv`` together with ``trades_raw.csv`` and
computes win rate and profit for both the model that was actually chosen and
all shadow models that were merely evaluated.  Results can optionally be
persisted to a JSON summary and used to update the bandit router's state file
so future model selection is biased toward higher performing models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------

def aggregate_decisions(
    decisions_file: Path, trades_file: Path, threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Return win rate and profit statistics for each model.

    Parameters
    ----------
    decisions_file : Path
        CSV log produced by :func:`LogDecision`.
    trades_file : Path
        CSV trade log containing ``decision_id`` and ``profit`` columns.
    threshold : float, default 0.5
        Probability threshold used to derive buy/sell actions for shadow
        models.
    """

    decisions = pd.read_csv(decisions_file, sep=";")
    decisions = decisions.rename(columns={"event_id": "decision_id"})
    trades = pd.read_csv(trades_file)
    merged = decisions.merge(trades[["decision_id", "profit"]], on="decision_id", how="left")
    merged["profit"] = merged["profit"].fillna(0.0)

    thr = float(threshold)

    def _pred_action(row: pd.Series) -> str:
        if int(row.get("chosen", 0)) == 1:
            return str(row.get("action", ""))
        prob = float(row.get("probability", 0.0))
        return "buy" if prob > thr else "sell"

    merged["pred_action"] = merged.apply(_pred_action, axis=1)
    merged["model_profit"] = np.where(
        merged["pred_action"] == merged["action"],
        merged["profit"],
        -merged["profit"],
    )
    merged["win"] = merged["model_profit"] > 0

    results: List[Dict[str, Any]] = []
    for model_idx, group in merged.groupby("model_idx"):
        chosen_g = group[group["chosen"] == 1]
        shadow_g = group[group["chosen"] == 0]

        def _stats(g: pd.DataFrame) -> Dict[str, Any]:
            trades = int(len(g))
            wins = int(g["win"].sum())
            profit = float(g["model_profit"].sum())
            win_rate = wins / trades if trades else 0.0
            return {
                "trades": trades,
                "wins": wins,
                "win_rate": win_rate,
                "profit": profit,
            }

        results.append(
            {
                "model_idx": int(model_idx),
                "chosen": _stats(chosen_g),
                "shadow": _stats(shadow_g),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Bandit refinement
# ---------------------------------------------------------------------------

def update_bandit_state(metrics: List[Dict[str, Any]], state_file: Path) -> None:
    """Update ``state_file`` with statistics from ``metrics``.

    Existing counts are incremented with the number of trades and wins for the
    chosen model of each entry.
    """

    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {"total": [], "wins": []}
    else:
        state = {"total": [], "wins": []}

    total = list(state.get("total", []))
    wins = list(state.get("wins", []))
    max_idx = max((m["model_idx"] for m in metrics), default=-1)
    if len(total) <= max_idx:
        total.extend([0] * (max_idx + 1 - len(total)))
        wins.extend([0] * (max_idx + 1 - len(wins)))

    for m in metrics:
        idx = m["model_idx"]
        total[idx] += m["chosen"]["trades"]
        wins[idx] += m["chosen"]["wins"]

    state_file.write_text(json.dumps({"total": total, "wins": wins}))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit meta strategy decisions")
    p.add_argument("--decisions", type=Path, required=True, help="Path to decisions.csv")
    p.add_argument("--trades", type=Path, required=True, help="Path to trades_raw.csv")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    p.add_argument("--out", type=Path, help="Optional JSON file to write summary")
    p.add_argument("--bandit-state", type=Path, help="Optional bandit state file to update")
    args = p.parse_args(argv)

    metrics = aggregate_decisions(args.decisions, args.trades, args.threshold)
    if args.out is not None:
        args.out.write_text(json.dumps(metrics, indent=2))
    if args.bandit_state is not None:
        update_bandit_state(metrics, args.bandit_state)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
