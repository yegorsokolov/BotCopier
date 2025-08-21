import json
import sys
import json
import json
import sys
from pathlib import Path

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


def _create_model(dir_path: Path, name: str, coeff: float) -> Path:
    model_dir = dir_path / name
    model_dir.mkdir()
    model_file = model_dir / f"{name}.json"
    model_file.write_text(json.dumps({"coefficients": [coeff], "threshold": 0.0}))
    return model_file


def _create_model_risk(dir_path: Path, name: str, expected: float, downside: float, coeff: float) -> Path:
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


def test_promote_by_sharpe(tmp_path: Path):
    tick_file = _create_tick_file(tmp_path)
    m1 = _create_model(tmp_path, "model_a", 1.0)
    m2 = _create_model(tmp_path, "model_b", -1.0)
    run_backtest(m1, tick_file)
    run_backtest(m2, tick_file)

    best_dir = tmp_path / "best"
    promote(tmp_path, best_dir, max_models=1, metric="sharpe_ratio")

    assert (best_dir / m1.name).exists()


def test_promote_by_sortino(tmp_path: Path):
    m1 = _create_model(tmp_path, "model_a", 1.0)
    m2 = _create_model(tmp_path, "model_b", 1.0)
    (tmp_path / "model_a" / "evaluation.json").write_text(
        json.dumps({"sortino_ratio": 2.0})
    )
    (tmp_path / "model_b" / "evaluation.json").write_text(
        json.dumps({"sortino_ratio": 1.0})
    )

    best_dir = tmp_path / "best"
    promote(tmp_path, best_dir, max_models=1, metric="sortino_ratio")

    assert (best_dir / m1.name).exists()


def test_promote_risk_reward(tmp_path: Path):
    tick_file = _create_tick_file(tmp_path)
    m1 = _create_model_risk(tmp_path, "model_a", 5.0, 1.0, 1.0)
    m2 = _create_model_risk(tmp_path, "model_b", 4.0, 1.0, 1.0)
    run_backtest(m1, tick_file)
    run_backtest(m2, tick_file)

    best_dir = tmp_path / "best"
    promote(tmp_path, best_dir, max_models=1, metric="risk_reward")

    assert (best_dir / m1.name).exists()


def test_promote_requires_models(tmp_path: Path):
    best_dir = tmp_path / "best"
    with pytest.raises(ValueError):
        promote(tmp_path, best_dir, max_models=1, metric="sharpe_ratio")
