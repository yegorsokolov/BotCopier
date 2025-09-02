import csv
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.model_fitting import load_logs, scale_features


def _write_log(path: Path) -> None:
    fields = [
        "schema_version",
        "event_id",
        "event_time",
        "action",
        "ticket",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "profit",
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
    ]
    rows = [
        ["1", "1", "2024.01.01 00:00:00", "OPEN", "1", "EURUSD", "0", "0.1", "1.1", "1.0", "1.2", "0", "0", "", "0.1", "0", "100"],
        ["1", "2", "2024.01.01 00:01:00", "OPEN", "2", "EURUSD", "0", "0.1", "1.1", "1.0", "1.2", "0", "0", "", "0.1", "0", "100"],
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_load_logs_basic(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_log(log_dir / "trades_1.csv")
    df, commits, checksums = load_logs(log_dir)
    assert len(df) == 2
    assert commits == []
    assert checksums == []
    assert "event_time" in df.columns


def test_scale_features() -> None:
    scaler = StandardScaler()
    X = np.array([[0.0], [1.0], [2.0]])
    Xs = scale_features(scaler, X)
    assert Xs.shape == X.shape
    assert np.allclose(Xs.mean(), 0.0, atol=1e-7)
