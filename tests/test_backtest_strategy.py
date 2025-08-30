from pathlib import Path
import csv
import json
import sys

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
    check_performance,
    analyze_shadow_trades,
)


FIXTURE = Path(__file__).parent / "fixtures" / "trades_small.csv"


def test_backtest_engine(tmp_path: Path):
    params_file = tmp_path / "params.json"
    params_file.write_text(
        json.dumps(
            {
                "coefficients": [1.0],
                "threshold": 0.1,
                "slippage": 0.01,
                "fee": 0.02,
                "magic": 42,
            }
        )
    )

    params = load_strategy_params(params_file)
    ticks = load_ticks(FIXTURE)
    metrics = backtest(
        ticks,
        params["coefficients"],
        params["threshold"],
        params["slippage"],
        params["fee"],
    )

    assert metrics["trade_count"] == 3
    assert metrics["win_rate"] == pytest.approx(2 / 3)
    assert metrics["profit_factor"] == pytest.approx(8.0)
    assert metrics["drawdown"] == pytest.approx(0.04)
    assert metrics["avg_latency"] == pytest.approx((10 + 15 + 12) / 3)

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

    result = run_backtest(params_file, FIXTURE)
    assert result["trade_count"] == 3

    # performance check passes for relaxed thresholds
    check_performance(result, min_win_rate=0.5, min_profit_factor=1.0)

    # and fails for demanding ones
    with pytest.raises(ValueError):
        check_performance(result, min_win_rate=0.9, min_profit_factor=10)


def test_analyze_shadow_trades(tmp_path: Path):
    path = tmp_path / "shadow_trades.csv"
    path.write_text(
        "timestamp;model_idx;result;profit\n"
        "2024.01.01 00:00;0;tp;10\n"
        "2024.01.01 00:01;0;sl;-5\n"
        "2024.01.01 00:02;1;tp;8\n"
        "2024.01.01 00:03;1;sl;-4\n"
        "2024.01.01 00:04;1;tp;6\n"
    )
    stats = analyze_shadow_trades(path)
    assert stats[0]["accuracy"] == 0.5
    assert stats[0]["profit"] == 5
    assert stats[1]["accuracy"] == pytest.approx(2 / 3)
    assert stats[1]["profit"] == pytest.approx(10)
