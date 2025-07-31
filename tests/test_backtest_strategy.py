from pathlib import Path
import sys
import csv
import json
import pytest

# add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.backtest_strategy import (
    load_strategy_params,
    load_ticks,
    backtest,
    write_report,
    update_metrics_csv,
    run_backtest,
)


def _write_mq4(file: Path) -> None:
    file.write_text("extern int MagicNumber = 7;\n double ModelThreshold = 0.0001;\n")


def _write_ticks(file: Path) -> None:
    with open(file, "w") as f:
        f.write("time;bid;ask;last;volume\n")
        f.write("2024.01.01 00:00:00;1.1000;1.1002;0;0\n")
        f.write("2024.01.01 00:00:01;1.1003;1.1005;0;0\n")
        f.write("2024.01.01 00:00:02;1.1001;1.1003;0;0\n")
        f.write("2024.01.01 00:00:03;1.1004;1.1006;0;0\n")


def test_backtest_basic(tmp_path: Path):
    mq4 = tmp_path / "strat.mq4"
    ticks_file = tmp_path / "ticks.csv"
    _write_mq4(mq4)
    _write_ticks(ticks_file)

    params = load_strategy_params(mq4)
    ticks = load_ticks(ticks_file)
    metrics = backtest(ticks, params.get("threshold", 0.0))

    assert metrics["trade_count"] == 3
    assert metrics["win_rate"] == pytest.approx(2 / 3)
    assert metrics["profit_factor"] == pytest.approx(3.0)

    report = tmp_path / "report.json"
    write_report(metrics, report)
    assert report.exists()
    with open(report) as f:
        data = json.load(f)
    assert data["trade_count"] == 3

    metrics_file = tmp_path / "metrics.csv"
    update_metrics_csv(metrics, metrics_file, params.get("magic", 0))
    with open(metrics_file, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)
    assert rows[0][0] == "time"
    assert rows[1][1] == str(params.get("magic", 0))

    result = run_backtest(mq4, ticks_file)
    assert result["trade_count"] == 3
