import json
from pathlib import Path

from scripts.batch_backtest import batch_backtest


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


def test_batch_backtest_promotes_best(tmp_path: Path):
    tick_file = _create_tick_file(tmp_path)
    _create_model(tmp_path, "model_a", 1.0)
    _create_model(tmp_path, "model_b", -1.0)

    summary_csv = tmp_path / "summary.csv"
    summary_json = tmp_path / "summary.json"
    best_dir = tmp_path / "best"

    results = batch_backtest(
        tmp_path,
        tick_file,
        summary_csv,
        summary_json,
        best_dir=best_dir,
        top_n=1,
        metric="sharpe_ratio",
    )

    assert len(results) == 2
    data = json.loads(summary_json.read_text())
    assert data[0]["model"].endswith(".json")
    assert (best_dir / "model_a.json").exists()
    assert not (best_dir / "model_b.json").exists()
