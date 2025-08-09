import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.auto_retrain import retrain_if_needed


def _write_metrics(file: Path, win_rate: float, drawdown: float) -> None:
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            ["time", "magic", "win_rate", "avg_profit", "trade_count", "drawdown", "sharpe"]
        )
        writer.writerow(["2024.01.01 00:00", "0", str(win_rate), "1.0", "10", str(drawdown), "0.0"])


def _write_features(file: Path, values: list[float]) -> None:
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["f1"])
        for v in values:
            writer.writerow([v])


def test_retrain_trigger(monkeypatch, tmp_path: Path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    files_dir = tmp_path / "files"
    for d in [log_dir, out_dir, files_dir]:
        d.mkdir()
    metrics_file = log_dir / "metrics.csv"
    _write_metrics(metrics_file, 0.3, 0.1)
    last_id_file = out_dir / "last_event_id"
    last_id_file.write_text("5")

    called = {}
    import scripts.train_target_clone as tc

    def fake_train(ld, od, incremental=True):
        called["train"] = (ld, od, incremental, tc.START_EVENT_ID)
        (od / "model.json").write_text(json.dumps({"last_event_id": 8}))

    def fake_publish(mf, fd):
        called["publish"] = (mf, fd)

    def fake_backtest(params_file, tick_file):
        called["backtest"] = (params_file, tick_file)
        return {"win_rate": 0.8, "drawdown": 0.05}

    monkeypatch.setattr("scripts.auto_retrain.train_model", fake_train)
    monkeypatch.setattr("scripts.auto_retrain.publish", fake_publish)
    monkeypatch.setattr("scripts.auto_retrain.run_backtest", fake_backtest)

    result = retrain_if_needed(log_dir, out_dir, files_dir)

    assert result is True
    assert called.get("train") == (log_dir, out_dir, True, 5)
    assert called.get("publish") == (out_dir / "model.json", files_dir)
    assert called.get("backtest") == (out_dir / "model.json", log_dir / "trades_raw.csv")
    assert last_id_file.read_text() == "8"


def test_retrain_not_triggered(monkeypatch, tmp_path: Path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    files_dir = tmp_path / "files"
    for d in [log_dir, out_dir, files_dir]:
        d.mkdir()
    metrics_file = log_dir / "metrics.csv"
    _write_metrics(metrics_file, 0.7, 0.1)

    called = {}

    def fake_train(*args, **kwargs):
        called["train"] = True

    def fake_publish(*args, **kwargs):
        called["publish"] = True

    def fake_backtest(*args, **kwargs):
        called["backtest"] = True
        return {"win_rate": 0.7, "drawdown": 0.1}

    monkeypatch.setattr("scripts.auto_retrain.train_model", fake_train)
    monkeypatch.setattr("scripts.auto_retrain.publish", fake_publish)
    monkeypatch.setattr("scripts.auto_retrain.run_backtest", fake_backtest)

    result = retrain_if_needed(log_dir, out_dir, files_dir)

    assert result is False
    assert called == {}


def test_retrain_no_improvement(monkeypatch, tmp_path: Path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    files_dir = tmp_path / "files"
    for d in [log_dir, out_dir, files_dir]:
        d.mkdir()
    metrics_file = log_dir / "metrics.csv"
    _write_metrics(metrics_file, 0.3, 0.1)

    called = {}

    def fake_train(ld, od, incremental=True):
        called["train"] = True
        (od / "model.json").write_text(json.dumps({"last_event_id": 1}))

    def fake_publish(*args, **kwargs):
        called["publish"] = True

    def fake_backtest(params_file, tick_file):
        called["backtest"] = True
        return {"win_rate": 0.2, "drawdown": 0.2}

    monkeypatch.setattr("scripts.auto_retrain.train_model", fake_train)
    monkeypatch.setattr("scripts.auto_retrain.publish", fake_publish)
    monkeypatch.setattr("scripts.auto_retrain.run_backtest", fake_backtest)

    result = retrain_if_needed(log_dir, out_dir, files_dir)

    assert result is False
    assert called.get("train") is True
    assert called.get("publish") is None
    assert called.get("backtest") is True


def test_retrain_prefers_onnx(monkeypatch, tmp_path: Path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    files_dir = tmp_path / "files"
    for d in [log_dir, out_dir, files_dir]:
        d.mkdir()
    metrics_file = log_dir / "metrics.csv"
    _write_metrics(metrics_file, 0.3, 0.1)
    last_id_file = out_dir / "last_event_id"
    last_id_file.write_text("5")

    called = {}

    def fake_train(ld, od, incremental=True):
        (od / "model.json").write_text(json.dumps({"last_event_id": 8}))
        (od / "model.onnx").write_bytes(b"x")

    def fake_publish(mf, fd):
        called["publish"] = mf

    def fake_backtest(params_file, tick_file):
        return {"win_rate": 0.8, "drawdown": 0.05}

    monkeypatch.setattr("scripts.auto_retrain.train_model", fake_train)
    monkeypatch.setattr("scripts.auto_retrain.publish", fake_publish)
    monkeypatch.setattr("scripts.auto_retrain.run_backtest", fake_backtest)

    result = retrain_if_needed(log_dir, out_dir, files_dir)

    assert result is True
    assert called.get("publish") == out_dir / "model.onnx"


def test_drift_triggers_retrain(monkeypatch, tmp_path: Path):
    log_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    files_dir = tmp_path / "files"
    for d in [log_dir, out_dir, files_dir]:
        d.mkdir()
    metrics_file = log_dir / "metrics.csv"
    # metrics are good so drift is sole trigger
    _write_metrics(metrics_file, 0.7, 0.1)
    baseline = log_dir / "baseline.csv"
    recent = log_dir / "recent.csv"
    _write_features(baseline, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    _write_features(recent, [10] * 10)

    called = {}

    def fake_train(ld, od, incremental=True):
        called["train"] = True
        (od / "model.json").write_text(json.dumps({}))

    def fake_publish(mf, fd):
        called["publish"] = True

    def fake_backtest(params_file, tick_file):
        called["backtest"] = True
        return {"win_rate": 0.8, "drawdown": 0.05}

    monkeypatch.setattr("scripts.auto_retrain.train_model", fake_train)
    monkeypatch.setattr("scripts.auto_retrain.publish", fake_publish)
    monkeypatch.setattr("scripts.auto_retrain.run_backtest", fake_backtest)

    result = retrain_if_needed(
        log_dir,
        out_dir,
        files_dir,
        baseline_file=baseline,
        recent_file=recent,
        drift_threshold=0.1,
    )

    assert result is True
    assert called.get("train") is True
    data = json.loads((out_dir / "model.json").read_text())
    assert "drift_metric" in data
