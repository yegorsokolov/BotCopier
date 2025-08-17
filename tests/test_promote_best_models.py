import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.promote_best_models import promote
from scripts.backtest_strategy import run_backtest


def _create_tick_file(dir_path: Path) -> Path:
    tick_file = dir_path / "ticks.csv"
    tick_file.write_text(
        "time;bid;ask;latency;f0\n"
        "0;1.0;1.1;0;1\n"
        "1;1.3;1.4;0;1\n"
        "2;1.2;1.3;0;1\n"
        "3;1.5;1.6;0;1\n"
    )
    return tick_file


def _create_model(dir_path: Path, name: str, metric_value: float, coeff: float) -> Path:
    model_dir = dir_path / name
    model_dir.mkdir()
    model_file = model_dir / f"{name}.json"
    model_file.write_text(json.dumps({"coefficients": [coeff], "threshold": 0.0}))
    with open(model_dir / "evaluation.json", "w") as f:
        json.dump({"accuracy": metric_value}, f)
    return model_file


def _create_model_risk(
    dir_path: Path, name: str, expected: float, downside: float, coeff: float
) -> Path:
    model_dir = dir_path / name
    model_dir.mkdir()
    model_file = model_dir / f"{name}.json"
    model_file.write_text(
        json.dumps(
            {
                "expected_return": expected,
                "downside_risk": downside,
                "coefficients": [coeff],
                "threshold": 0.0,
            }
        )
    )
    return model_file


def test_promote_uses_backtest_metric(tmp_path: Path):
    tick_file = _create_tick_file(tmp_path)
    m1 = _create_model(tmp_path, "model_a", 0.9, 1.0)
    m2 = _create_model(tmp_path, "model_b", 0.9, -1.0)
    run_backtest(m1, tick_file)
    run_backtest(m2, tick_file)

    best_dir = tmp_path / "best"
    promote(
        tmp_path,
        best_dir,
        max_models=1,
        metric="accuracy",
        backtest_metric="sharpe",
    )

    assert (best_dir / m1.name).exists()


def test_promote_risk_reward(tmp_path: Path):
    tick_file = _create_tick_file(tmp_path)
    m1 = _create_model_risk(tmp_path, "model_a", 5.0, 1.0, 1.0)
    m2 = _create_model_risk(tmp_path, "model_b", 4.0, 1.0, 1.0)
    run_backtest(m1, tick_file)
    run_backtest(m2, tick_file)

    best_dir = tmp_path / "best"
    promote(
        tmp_path,
        best_dir,
        max_models=1,
        metric="risk_reward",
        backtest_metric="sharpe",
    )

    assert (best_dir / m1.name).exists()


def test_promote_backtest_threshold(tmp_path: Path):
    tick_file = _create_tick_file(tmp_path)
    m = _create_model(tmp_path, "model_a", 0.9, -1.0)
    run_backtest(m, tick_file)

    best_dir = tmp_path / "best"
    with pytest.raises(ValueError):
        promote(
            tmp_path,
            best_dir,
            max_models=1,
            metric="accuracy",
            backtest_metric="sharpe",
            min_backtest=0.0,
        )

