import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pytest
pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("optuna")
pytest.importorskip("scipy")

from botcopier.training.pipeline import train


_row_zero = st.fixed_dictionaries(
    {
        "label": st.just(0),
        "price": st.floats(
            min_value=0.5,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        "volume": st.integers(min_value=50, max_value=500),
        "spread": st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        "hour": st.integers(min_value=0, max_value=23),
        "symbol": st.sampled_from(["EURUSD", "GBPUSD"]),
    }
)

_row_one = st.fixed_dictionaries(
    {
        "label": st.just(1),
        "price": st.floats(
            min_value=0.5,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        "volume": st.integers(min_value=50, max_value=500),
        "spread": st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        "hour": st.integers(min_value=0, max_value=23),
        "symbol": st.sampled_from(["EURUSD", "GBPUSD"]),
    }
)


def _write_dataset(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = ["label,price,volume,spread,hour,symbol"]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["label"]),
                    f"{float(row['price']):.6f}",
                    str(int(row["volume"])),
                    f"{float(row['spread']):.6f}",
                    str(int(row["hour"])),
                    str(row["symbol"]),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n")


@settings(max_examples=3, deadline=None)
@given(
    zeros=st.lists(_row_zero, min_size=2, max_size=5),
    ones=st.lists(_row_one, min_size=2, max_size=5),
)
def test_training_pipeline_records_artifacts(tmp_path: Path, zeros, ones) -> None:
    rows: list[Dict[str, object]] = []
    for idx in range(max(len(zeros), len(ones))):
        if idx < len(zeros):
            rows.append(zeros[idx])
        if idx < len(ones):
            rows.append(ones[idx])
    data_file = tmp_path / "trades_raw.csv"
    _write_dataset(data_file, rows)

    out_dir = tmp_path / "out"
    config_snapshot = {
        "data": {"data": str(data_file)},
        "training": {"n_splits": 2, "lite_mode": True},
        "execution": {},
    }
    snapshot_digest = hashlib.sha256(
        json.dumps(config_snapshot, sort_keys=True).encode("utf-8")
    ).hexdigest()

    train(
        data_file,
        out_dir,
        n_splits=2,
        cv_gap=1,
        param_grid=[{}],
        lite_mode=True,
        config_hash=snapshot_digest,
        config_snapshot=config_snapshot,
    )

    model = json.loads((out_dir / "model.json").read_text())
    metadata = model["metadata"]

    key = str(data_file.resolve())
    expected_hash = hashlib.sha256(data_file.read_bytes()).hexdigest()
    assert model["data_hashes"][key] == expected_hash
    assert metadata["config_hash"] == snapshot_digest
    assert metadata["config_snapshot_hash"] == snapshot_digest
    assert metadata["config_snapshot"] == config_snapshot
    snapshot_path = out_dir / metadata["config_snapshot_path"]
    assert snapshot_path.exists()
    assert json.loads(snapshot_path.read_text()) == config_snapshot

    deps_path = out_dir / metadata["dependencies_file"]
    assert deps_path.exists()
    assert metadata["dependencies_hash"] == hashlib.sha256(
        deps_path.read_bytes()
    ).hexdigest()
    assert deps_path.read_text().strip()

    hashes_file = out_dir / metadata["data_hashes_path"]
    assert hashes_file.exists()
    recorded_hashes = json.loads(hashes_file.read_text())
    assert recorded_hashes[key] == expected_hash
