import json
import hashlib
import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("optuna")
pytest.importorskip("scipy")

from botcopier.scripts.evaluation import evaluate
from botcopier.training.pipeline import train
from scripts.promote_strategy import promote


def _write_training_logs(path: Path) -> None:
    rows = [
        "label,price,volume,spread,hour,symbol",
        "1,1.0,100,1.0,0,EURUSD",
        "0,1.1,110,1.1,1,EURUSD",
        "1,1.2,120,1.2,2,EURUSD",
        "0,1.3,130,1.3,3,EURUSD",
    ]
    path.write_text("\n".join(rows) + "\n")


def _write_predictions_csv(path: Path) -> None:
    rows = [
        "timestamp;symbol;direction;lots;probability;value;log_variance;executed_model_idx;decision_id",
        "2024.01.01 00:00;EURUSD;buy;0.10;0.80;15;0.0;0;1",
        "2024.01.01 01:00;EURUSD;sell;0.10;0.30;-10;0.0;0;2",
    ]
    path.write_text("\n".join(rows) + "\n")


def _write_trade_log_csv(path: Path) -> None:
    rows = [
        "schema_version;event_id;event_time;action;ticket;symbol;order_type;lots;profit;decision_id",
        "1;1;2024.01.01 00:00;OPEN;10;EURUSD;0;0.10;0;1",
        "1;2;2024.01.01 00:10;CLOSE;10;EURUSD;0;0.10;15;1",
        "1;3;2024.01.01 01:00;OPEN;11;EURUSD;1;0.10;0;2",
        "1;4;2024.01.01 01:15;CLOSE;11;EURUSD;1;0.10;-10;2",
    ]
    path.write_text("\n".join(rows) + "\n")


@pytest.mark.integration
def test_full_pipeline(tmp_path: Path) -> None:
    """End-to-end training pipeline produces model artifacts."""
    data_file = tmp_path / "trades_raw.csv"
    _write_training_logs(data_file)

    # Train with minimal cross-validation to keep runtime low
    out_dir = tmp_path / "out"
    train(
        data_file,
        out_dir,
        n_splits=2,
        cv_gap=1,
        param_grid=[{}],
        lite_mode=True,
    )

    model_path = out_dir / "model.json"
    assert model_path.exists()
    model = json.loads(model_path.read_text())
    for field in ("coefficients", "intercept", "feature_names"):
        assert field in model

    expected_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
    key = str(data_file.resolve())
    assert model["data_hashes"][key] == expected_hash
    assert "risk_metrics" in model
    assert set(model["risk_metrics"].keys()) >= {"max_drawdown", "var_95"}


@pytest.mark.integration
def test_train_promote_and_verify(tmp_path: Path) -> None:
    """Train a strategy then promote and verify it on synthetic data."""
    data_file = tmp_path / "trades_raw.csv"
    _write_training_logs(data_file)

    shadow = tmp_path / "shadow" / "modelA"
    train(
        data_file,
        shadow,
        n_splits=2,
        cv_gap=1,
        param_grid=[{}],
        lite_mode=True,
    )

    # Synthetic performance metrics for promotion
    (shadow / "oos.csv").write_text("0.1\n0.2\n-0.1\n")
    (shadow / "orders.csv").write_text("market\nmarket\nlimit\n")

    live = tmp_path / "live"
    metrics_dir = tmp_path / "metrics"
    registry = tmp_path / "models" / "active.json"

    promote(shadow.parent, live, metrics_dir, registry, max_drawdown=0.5, max_risk=1.0)

    promoted = live / "modelA"
    assert promoted.exists()

    report = json.loads((metrics_dir / "risk.json").read_text())
    assert "modelA" in report
    reg = json.loads(registry.read_text())
    assert reg["modelA"] == str(promoted)

    predictions = tmp_path / "predictions.csv"
    trades = tmp_path / "trade_log.csv"
    _write_predictions_csv(predictions)
    _write_trade_log_csv(trades)

    stats = evaluate(
        predictions,
        trades,
        window=60,
        model_json=promoted / "model.json",
        fee_per_trade=0.0,
        slippage_bps=0.0,
    )

    assert stats["matched_events"] == 2
    assert stats["predicted_events"] == 2
    assert stats["actual_events"] == 2
    assert stats["precision"] == pytest.approx(1.0)
    assert stats["accuracy"] == pytest.approx(1.0)
    assert stats["profit_factor"] == pytest.approx(1.5)
    assert pytest.approx(stats["gross_profit"], rel=1e-6) == 15.0
    assert pytest.approx(stats["gross_loss"], rel=1e-6) == 10.0
    assert math.isfinite(stats["expected_return_net"])
    assert math.isfinite(stats["brier_score"])
