import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score


def _parse_time(value: str) -> datetime:
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    raise ValueError(f"Unrecognised time format: {value}")


def _load_predictions(pred_file: Path) -> List[Dict]:
    with open(pred_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        preds = []
        for r in reader:
            ts = _parse_time(
                r.get("timestamp") or r.get("time") or r[reader.fieldnames[0]]
            )
            direction_raw = str(
                r.get("direction") or r.get("order_type") or ""
            ).strip().lower()
            if direction_raw in ("1", "buy"):
                direction = 1
            elif direction_raw in ("0", "-1", "sell"):
                direction = -1
            else:
                direction = 1
            preds.append(
                {
                    "timestamp": ts,
                    "symbol": r.get("symbol", ""),
                    "direction": direction,
                    "lots": float(r.get("lots", 0) or 0),
                    "probability": (
                        float(r.get("probability") or r.get("prob") or r.get("proba"))
                        if (
                            r.get("probability")
                            or r.get("prob")
                            or r.get("proba")
                        )
                        else None
                    ),
                    "executed_model_idx": (
                        int(float(r.get("executed_model_idx") or r.get("model_idx") or r.get("model") or -1))
                        if (r.get("executed_model_idx") or r.get("model_idx") or r.get("model"))
                        else None
                    ),
                    "decision_id": (
                        int(float(r.get("decision_id"))) if r.get("decision_id") else None
                    ),
                }
            )
    return preds


def _load_actual_trades(log_file: Path) -> List[Dict]:
    with open(log_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        open_map: Dict[str, Dict] = {}
        trades = []
        for r in reader:
            action = (r.get("action") or "").upper()
            ticket = r.get("ticket")
            ts = _parse_time(
                r.get("event_time")
                or r.get("time_event")
                or r[reader.fieldnames[0]]
            )
            if action == "OPEN":
                open_map[ticket] = {
                    "open_time": ts,
                    "symbol": r.get("symbol", ""),
                    "direction": 1
                    if int(float(r.get("order_type", 0))) == 0
                    else -1,
                    "lots": float(r.get("lots", 0) or 0),
                    "executed_model_idx": (
                        int(float(r.get("executed_model_idx", -1) or -1))
                    ),
                    "decision_id": (
                        int(float(r.get("decision_id", 0) or 0))
                    ),
                }
            elif action == "CLOSE" and ticket in open_map:
                o = open_map.pop(ticket)
                profit = float(r.get("profit", 0) or 0)
                trade = {**o, "close_time": ts, "profit": profit}
                trades.append(trade)
    return trades


def evaluate(pred_file: Path, actual_log: Path, window: int, model_json: Optional[Path] = None) -> Dict:
    predictions = _load_predictions(pred_file)
    actual_trades = _load_actual_trades(actual_log)
    conformal_lower = None
    conformal_upper = None
    model_value_mean = None
    model_value_std = None
    model_value_atoms = None
    model_value_dist = None
    model_value_quantiles = None
    if model_json is not None:
        try:
            with open(model_json) as f:
                m = json.load(f)
            conformal_lower = m.get("conformal_lower")
            conformal_upper = m.get("conformal_upper")
            model_value_mean = m.get("value_mean")
            model_value_std = m.get("value_std")
            model_value_atoms = m.get("value_atoms")
            model_value_dist = m.get("value_distribution")
            model_value_quantiles = m.get("value_quantiles")
        except Exception:
            pass
    matches = 0
    gross_profit = 0.0
    gross_loss = 0.0
    profits: List[float] = []
    used = set()
    predictions_per_model: Dict[int, int] = {}
    matches_per_model: Dict[int, int] = {}
    trade_by_decision: Dict[int, Dict] = {}
    bound_in = 0
    bound_total = 0
    for idx, trade in enumerate(actual_trades):
        did = trade.get("decision_id")
        if did is not None and did not in trade_by_decision:
            trade_by_decision[did] = {"index": idx, "trade": trade}
    for pred in predictions:
        model_idx = pred.get("executed_model_idx")
        if model_idx is not None:
            predictions_per_model[model_idx] = predictions_per_model.get(model_idx, 0) + 1
        match_idx = None
        if pred.get("decision_id") is not None:
            entry = trade_by_decision.get(pred["decision_id"])
            if entry and entry["index"] not in used:
                match_idx = entry["index"]
        if match_idx is None:
            for idx, trade in enumerate(actual_trades):
                if idx in used:
                    continue
                if trade["symbol"] != pred["symbol"]:
                    continue
                if trade["direction"] != pred["direction"]:
                    continue
                delta = (trade["open_time"] - pred["timestamp"]).total_seconds()
                if 0 <= delta <= window:
                    match_idx = idx
                    break
        if match_idx is not None:
            used.add(match_idx)
            matches += 1
            trade = actual_trades[match_idx]
            p = trade["profit"]
            profits.append(p)
            if p >= 0:
                gross_profit += p
            else:
                gross_loss += -p
            if model_idx is not None:
                matches_per_model[model_idx] = matches_per_model.get(model_idx, 0) + 1
            if conformal_lower is not None and conformal_upper is not None:
                prob = pred.get("probability")
                if prob is not None:
                    bound_total += 1
                    if conformal_lower <= prob <= conformal_upper:
                        bound_in += 1
    tp = matches
    fp = len(predictions) - matches
    fn = len(actual_trades) - matches
    total = tp + fp + fn
    accuracy = tp / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss else float("inf")
    expectancy = (gross_profit - gross_loss) / matches if matches else 0.0
    expected_return = sum(profits) / len(profits) if profits else 0.0
    downside = [p for p in profits if p < 0]
    downside_risk = -sum(downside) / len(downside) if downside else 0.0
    risk_reward = expected_return - downside_risk
    sharpe = 0.0
    sortino = 0.0
    if len(profits) > 1:
        mean = expected_return
        variance = sum((p - mean) ** 2 for p in profits) / (len(profits) - 1)
        std = math.sqrt(variance)
        if std > 0:
            sharpe = mean / std
        downside_dev = 0.0
        if downside:
            downside_dev = math.sqrt(sum((p) ** 2 for p in downside) / len(downside))
        if downside_dev > 0:
            sortino = mean / downside_dev
    conformal = bound_in / bound_total if bound_total else None
    stats: Dict[str, object] = {
        "matched_events": matches,
        "predicted_events": len(predictions),
        "actual_events": len(actual_trades),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "expectancy": expectancy,
        "expected_return": expected_return,
        "downside_risk": downside_risk,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "risk_reward": risk_reward,
        "conformal_coverage": conformal,
        "predictions_per_model": predictions_per_model,
        "matches_per_model": matches_per_model,
    }
    if model_value_mean is not None:
        stats["model_value_mean"] = model_value_mean
    if model_value_std is not None:
        stats["model_value_std"] = model_value_std
    if model_value_atoms and model_value_dist:
        stats["model_value_atoms"] = model_value_atoms
        stats["model_value_distribution"] = model_value_dist
    if model_value_quantiles:
        stats["model_value_quantiles"] = model_value_quantiles
    return stats


def evaluate_model(model, X, y) -> float:
    preds = model.predict(X)
    return accuracy_score(y, preds)

