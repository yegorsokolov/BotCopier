import json
import hashlib
from pathlib import Path

import pytest

from botcopier.training.pipeline import train


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
