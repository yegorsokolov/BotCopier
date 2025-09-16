import csv
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip("numpy")
import numpy as np

from botcopier.scripts.evaluation import search_decision_threshold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation import evaluate


def _write_prediction(
    file: Path, ts: str, direction: str = "buy", probability: float = 0.9
):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["timestamp", "symbol", "direction", "lots", "probability"])
        writer.writerow([ts, "EURUSD", direction, "0.1", probability])


def _write_actual(file: Path, ts_open: str, ts_close: str, order_type: str = "0"):
    fields = [
        "event_time",
        "action",
        "ticket",
        "symbol",
        "order_type",
        "lots",
        "profit",
    ]
    rows = [
        [ts_open, "OPEN", "1", "EURUSD", order_type, "0.1", "0"],
        [ts_close, "CLOSE", "1", "EURUSD", order_type, "0.1", "10"],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_evaluate(tmp_path: Path):
    pred_file = tmp_path / "preds.csv"
    log_file = tmp_path / "trades.csv"
    _write_prediction(pred_file, "2024.01.01 00:00:00", probability=0.9)
    with open(pred_file, "a", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["2024.01.02 00:00:00", "EURUSD", "buy", "0.1", 0.1])
    _write_actual(log_file, "2024.01.01 00:00:05", "2024.01.01 00:01:00")

    model_file = tmp_path / "model.json"
    model_info = {
        "value_mean": 1.0,
        "value_std": 0.5,
        "value_atoms": [-1, 1],
        "value_distribution": [0.4, 0.6],
    }
    with open(model_file, "w") as f:
        json.dump(model_info, f)

    stats = evaluate(pred_file, log_file, window=60, model_json=model_file)

    assert stats["matched_events"] == 1
    assert stats["predicted_events"] == 2
    assert stats["accuracy"] == 0.5
    assert stats["precision"] == 0.5
    assert stats["recall"] == 1.0
    assert stats["profit_factor"] == float("inf")
    assert stats["sharpe_ratio"] == 0.0
    assert stats["sortino_ratio"] == 0.0
    assert stats["expectancy"] == 10
    assert stats["expected_return"] == 10
    assert stats["downside_risk"] == 0
    assert stats["risk_reward"] == 10
    assert stats["var_95"] == 10
    assert stats["es_95"] == 10
    assert stats["model_value_mean"] == model_info["value_mean"]
    assert stats["model_value_std"] == model_info["value_std"]
    assert stats["model_value_atoms"] == model_info["value_atoms"]
    assert stats["model_value_distribution"] == model_info["value_distribution"]
    assert stats["roc_auc"] == 1.0
    assert stats["pr_auc"] == 1.0
    assert stats["brier_score"] == pytest.approx(0.01)
    assert stats["reliability_curve"]["prob_true"]


def test_search_decision_threshold_selects_best_profit() -> None:
    y = np.array([1, 0, 1, 0], dtype=float)
    probas = np.array([0.9, 0.6, 0.4, 0.2], dtype=float)
    profits = np.array([2.0, -3.0, 1.5, -0.5], dtype=float)

    threshold, metrics = search_decision_threshold(
        y, probas, profits, objective="profit"
    )

    assert threshold == pytest.approx(0.9)
    assert metrics["threshold_objective"] == "profit"
    assert metrics["profit"] == pytest.approx(2.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_search_decision_threshold_respects_risk_limits() -> None:
    y = np.array([1, 0, 1], dtype=float)
    probas = np.ones_like(y) * 0.9
    profits = np.array([-5.0, -4.0, -3.0], dtype=float)

    with pytest.raises(ValueError):
        search_decision_threshold(y, probas, profits, max_drawdown=1.0)


def test_direction_mapping(tmp_path: Path):
    cases = [
        ("1", "0"),
        ("buy", "0"),
        ("0", "1"),
        ("-1", "1"),
        ("sell", "1"),
        ("", "0"),
    ]

    for direction_value, order_type in cases:
        pred_file = tmp_path / f"pred_{direction_value or 'empty'}.csv"
        log_file = tmp_path / f"trades_{direction_value or 'empty'}.csv"
        _write_prediction(pred_file, "2024.01.01 00:00:00", direction_value)
        _write_actual(
            log_file,
            "2024.01.01 00:00:05",
            "2024.01.01 00:01:00",
            order_type,
        )

        stats = evaluate(pred_file, log_file, window=60)
        assert stats["matched_events"] == 1
