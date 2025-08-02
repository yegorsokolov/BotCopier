import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.auto_retrain import retrain_if_needed


def _write_metrics(file: Path, win_rate: float, sharpe: float = 0.0) -> None:
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "time",
            "magic",
            "win_rate",
            "avg_profit",
            "trade_count",
            "sharpe",
        ])
        writer.writerow(["2024.01.01 00:00", "0", str(win_rate), "1.0", "10", str(sharpe)])


def test_retrain_trigger(monkeypatch, tmp_path: Path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    files_dir = tmp_path / "files"
    log_dir.mkdir()
    out_dir.mkdir()
    files_dir.mkdir()
    metrics_file = log_dir / "metrics.csv"
    _write_metrics(metrics_file, 0.3)

    called = {}

    def fake_train(ld, od, incremental=True):
        called["train"] = (ld, od, incremental)

    def fake_publish(mf, fd):
        called["publish"] = (mf, fd)

    monkeypatch.setattr("scripts.auto_retrain.train", fake_train)
    monkeypatch.setattr("scripts.auto_retrain.publish", fake_publish)

    result = retrain_if_needed(log_dir, out_dir, files_dir, win_rate_threshold=0.4)

    assert result is True
    assert called.get("train") == (log_dir, out_dir, True)
    assert called.get("publish") == (out_dir / "model.json", files_dir)


def test_retrain_not_triggered(monkeypatch, tmp_path: Path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    files_dir = tmp_path / "files"
    log_dir.mkdir()
    out_dir.mkdir()
    files_dir.mkdir()
    metrics_file = log_dir / "metrics.csv"
    _write_metrics(metrics_file, 0.7)

    called = {}

    def fake_train(*args, **kwargs):
        called["train"] = True

    def fake_publish(*args, **kwargs):
        called["publish"] = True

    monkeypatch.setattr("scripts.auto_retrain.train", fake_train)
    monkeypatch.setattr("scripts.auto_retrain.publish", fake_publish)

    result = retrain_if_needed(log_dir, out_dir, files_dir, win_rate_threshold=0.4)

    assert result is False
    assert "train" not in called
    assert "publish" not in called


def test_retrain_trigger_sharpe(monkeypatch, tmp_path: Path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    files_dir = tmp_path / "files"
    log_dir.mkdir()
    out_dir.mkdir()
    files_dir.mkdir()
    metrics_file = log_dir / "metrics.csv"
    _write_metrics(metrics_file, 0.7, -0.2)

    called = {}

    def fake_train(ld, od, incremental=True):
        called["train"] = (ld, od, incremental)

    def fake_publish(mf, fd):
        called["publish"] = (mf, fd)

    monkeypatch.setattr("scripts.auto_retrain.train", fake_train)
    monkeypatch.setattr("scripts.auto_retrain.publish", fake_publish)

    result = retrain_if_needed(
        log_dir,
        out_dir,
        files_dir,
        win_rate_threshold=0.4,
        sharpe_threshold=0.0,
    )

    assert result is True
    assert called.get("train") == (log_dir, out_dir, True)
    assert called.get("publish") == (out_dir / "model.json", files_dir)
