import json
import hashlib
from pathlib import Path

import pytest

from botcopier.training.pipeline import train
from scripts.promote_strategy import promote


@pytest.mark.integration
def test_full_pipeline(tmp_path: Path) -> None:
    """End-to-end training pipeline produces model artifacts."""
    # Prepare synthetic training logs
    data_file = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "1,1.0,100,1.0,0,EURUSD\n",
        "0,1.1,110,1.1,1,EURUSD\n",
        "1,1.2,120,1.2,2,EURUSD\n",
        "0,1.3,130,1.3,3,EURUSD\n",
    ]
    data_file.write_text("".join(rows))

    # Train with minimal cross-validation to keep runtime low
    out_dir = tmp_path / "out"
    train(data_file, out_dir, n_splits=2, cv_gap=1, param_grid=[{}])

    model_path = out_dir / "model.json"
    assert model_path.exists()
    model = json.loads(model_path.read_text())
    for field in ("coefficients", "intercept", "feature_names"):
        assert field in model

    expected_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
    key = str(data_file.resolve())
    assert model["data_hashes"][key] == expected_hash


@pytest.mark.integration
def test_train_promote_and_verify(tmp_path: Path) -> None:
    """Train a strategy then promote and verify it on synthetic data."""
    data_file = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,volume,spread,hour,symbol\n",
        "1,1.0,100,1.0,0,EURUSD\n",
        "0,1.1,110,1.1,1,EURUSD\n",
        "1,1.2,120,1.2,2,EURUSD\n",
        "0,1.3,130,1.3,3,EURUSD\n",
    ]
    data_file.write_text("".join(rows))

    shadow = tmp_path / "shadow" / "modelA"
    train(data_file, shadow, n_splits=2, cv_gap=1, param_grid=[{}])

    # Synthetic performance metrics for promotion
    (shadow / "oos.csv").write_text("0.1\n0.2\n-0.1\n")
    (shadow / "orders.csv").write_text("market\nmarket\nmarket\n")

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
