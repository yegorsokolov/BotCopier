import csv
import json
import hashlib
from pathlib import Path

from scripts import train_target_clone as tc


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
        [
            "3",
            "2024.01.01 01:00:00",
            "",
            "",
            "OPEN",
            "2",
            "",
            "",
            "EURUSD",
            "1",
            "0.1",
            "1.2000",
            "1.1950",
            "1.2100",
            "0",
            "",
            "0.1",
        ],
        [
            "4",
            "2024.01.01 01:30:00",
            "",
            "",
            "CLOSE",
            "2",
            "",
            "",
            "EURUSD",
            "1",
            "0.1",
            "1.1950",
            "1.1950",
            "1.2100",
            "-5",
            "",
            "0",
        ],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_rl_refinement(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_sample.csv"
    _write_log(log_file)

    checksum = hashlib.sha256(log_file.read_bytes()).hexdigest()
    manifest = log_file.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"file": log_file.name, "checksum": checksum, "commit": "abc"}))

    # Force resource detection to allow RL run
    monkeypatch.setattr(tc, "HAS_SB3", True)
    monkeypatch.setattr(tc, "_has_sufficient_gpu", lambda min_gb=1.0: True)
    monkeypatch.setattr(tc, "_has_sufficient_ram", lambda min_gb=1.0: True)

    tc.train(data_dir, out_dir)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert "rl_steps" in data and "rl_reward" in data
