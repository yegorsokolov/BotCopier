import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd

from botcopier.data.loading import _load_logs
from botcopier.training.pipeline import train


def _write_log(path: Path, rows: int) -> None:
    df = pd.DataFrame(
        {
            "label": [0, 1] * (rows // 2),
            "spread": [1.0] * rows,
            "hour": [i % 24 for i in range(rows)],
            "volume": [100] * rows,
            "row_id": list(range(rows)),
        }
    )
    df.to_csv(path, index=False)


def test_load_logs_chunks_when_not_lite(tmp_path):
    csv = tmp_path / "trades_raw.csv"
    _write_log(csv, 120_000)
    chunks, feature_cols, _ = _load_logs(tmp_path, lite_mode=False, chunk_size=50_000)
    assert not isinstance(chunks, pd.DataFrame)
    assert "volume" in feature_cols
    sizes = [len(c) for c in chunks]
    assert sizes == [50_000, 50_000, 20_000]


def test_train_iterates_chunks(tmp_path):
    csv = tmp_path / "trades_raw.csv"
    _write_log(csv, 100_000)
    out_dir = tmp_path / "out"
    train(csv, out_dir, chunk_size=50_000, mode="standard")
    model = json.loads((out_dir / "model.json").read_text())
    assert model["mode"] == "standard"


def test_load_logs_in_memory_small_file(tmp_path):
    csv = tmp_path / "trades_raw.csv"
    _write_log(csv, 10)
    df, _, _ = _load_logs(tmp_path, lite_mode=False, mmap_threshold=10**9)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10


def test_load_logs_memory_map_large_file(tmp_path):
    csv = tmp_path / "trades_raw.csv"
    _write_log(csv, 10)
    chunks, _, _ = _load_logs(tmp_path, lite_mode=False, mmap_threshold=1)
    assert not isinstance(chunks, pd.DataFrame)
    sizes = [len(c) for c in chunks]
    assert sizes == [10]


def test_chunked_loader_produces_meta_labels(tmp_path):
    csv = tmp_path / "trades_raw.csv"
    df = pd.DataFrame(
        {
            "label": [1, 0, 1, 0],
            "price": [1.00, 0.95, 1.10, 1.05],
            "spread": [0.02, 0.03, 0.02, 0.03],
            "hour": [0, 1, 2, 3],
        }
    )
    df.to_csv(csv, index=False)

    chunks, _, _ = _load_logs(tmp_path, lite_mode=False, chunk_size=2)
    chunk_list = list(chunks)
    assert chunk_list
    required_cols = {"take_profit", "stop_loss", "horizon", "tp_time", "sl_time", "meta_label"}
    assert all(required_cols.issubset(chunk.columns) for chunk in chunk_list)
    assert any(chunk["meta_label"].notna().any() for chunk in chunk_list)
