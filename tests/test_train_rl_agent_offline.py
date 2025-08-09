import csv
import json
from pathlib import Path
import sys
import gzip

import pytest

try:  # optional dependency for decision transformer
    import torch  # type: ignore
    import transformers  # type: ignore
    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional
    HAS_TRANSFORMERS = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_rl_agent import train


def _write_log(file: Path) -> None:
    fields = [
        "event_id",
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "profit",
        "comment",
        "remaining_lots",
    ]
    rows = [
        [
            "1",
            "2024.01.01 00:00:00",
            "",
            "",
            "OPEN",
            "1",
            "",
            "",
            "EURUSD",
            "0",
            "0.1",
            "1.1000",
            "1.0950",
            "1.1100",
            "0",
            "",
            "0.1",
        ],
        [
            "2",
            "2024.01.01 00:30:00",
            "",
            "",
            "CLOSE",
            "1",
            "",
            "",
            "EURUSD",
            "0",
            "0.1",
            "1.1050",
            "1.0950",
            "1.1100",
            "5",
            "",
            "0",
        ],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_train_cql(tmp_path: Path) -> None:
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_log(data_dir / "trades_1.csv")

    train(data_dir, out_dir, algo="cql", training_steps=5)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("training_type") == "offline_rl"
    assert data.get("algo") == "cql"


def test_train_cql_compress_model(tmp_path: Path) -> None:
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_log(data_dir / "trades_1.csv")

    train(data_dir, out_dir, algo="cql", training_steps=5, compress_model=True)

    model_file = out_dir / "model.json.gz"
    assert model_file.exists()
    with gzip.open(model_file, "rt") as f:
        data = json.load(f)
    assert data.get("training_type") == "offline_rl"


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers required")
def test_train_decision_transformer(tmp_path: Path) -> None:
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_log(data_dir / "trades_1.csv")

    train(data_dir, out_dir, algo="decision_transformer", training_steps=1)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("algo") == "decision_transformer"
    assert "transformer_weights" in data
    assert data.get("sequence_length", 0) > 0
