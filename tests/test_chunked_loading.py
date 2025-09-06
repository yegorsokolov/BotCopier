import json
from pathlib import Path

import pandas as pd

from scripts.train_target_clone import _load_logs, train


def _write_log(path: Path, rows: int) -> None:
    df = pd.DataFrame(
        {
            "label": [0, 1] * (rows // 2),
            "spread": [1.0] * rows,
            "hour": [i % 24 for i in range(rows)],
            "volume": [100] * rows,
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
