import csv
import json
from pathlib import Path
import sys

# add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_logs import main


def _write_trades(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["decision_id", "profit", "entry_time", "exit_time", "slippage"])
        writer.writerow([1, 10, "2024-01-01 00:00:00", "2024-01-01 00:01:00", 0.1])
        writer.writerow([2, -5, "2024-01-01 00:01:00", "2024-01-01 00:03:00", -0.2])
        writer.writerow([3, 15, "2024-01-01 00:02:00", "2024-01-01 00:04:00", 0.0])


def _write_metrics(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["decision_id", "prediction"])
        writer.writerow([1, 1])
        writer.writerow([2, 0])
        writer.writerow([3, 0])


def _write_decisions(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "decision_id",
                "action",
                "model_idx",
                "executed_model_idx",
                "probability",
            ]
        )
        writer.writerow([1, "shadow", 0, 0, 0.4])
        writer.writerow([2, "buy", 0, 0, 0.7])


def test_session_summary_fields(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trades_file = logs_dir / "trades_raw.csv"
    metrics_file = logs_dir / "metrics.csv"
    _write_trades(trades_file)
    _write_metrics(metrics_file)
    decisions_file = logs_dir / "decisions.csv"
    _write_decisions(decisions_file)
    summary_file = tmp_path / "session_summary.json"
    summaries_csv = logs_dir / "summaries.csv"

    main(
        [
            "--trades-file",
            str(trades_file),
            "--metrics-file",
            str(metrics_file),
            "--summary-file",
            str(summary_file),
            "--summaries-file",
            str(summaries_csv),
            "--decisions-file",
            str(decisions_file),
            "--n-jobs",
            "1",
        ]
    )

    assert summary_file.exists()
    data = json.loads(summary_file.read_text())
    for key in [
        "win_rate",
        "sharpe",
        "avg_hold_time",
        "slippage_mean",
        "slippage_std",
        "prediction_accuracy",
    ]:
        assert key in data
    assert summaries_csv.exists()


def test_summary_deterministic_n_jobs(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    trades_file = logs_dir / "trades_raw.csv"
    metrics_file = logs_dir / "metrics.csv"
    decisions_file = logs_dir / "decisions.csv"
    _write_trades(trades_file)
    _write_metrics(metrics_file)
    _write_decisions(decisions_file)
    summary1 = tmp_path / "sum1.json"
    summary2 = tmp_path / "sum2.json"

    main(
        [
            "--trades-file",
            str(trades_file),
            "--metrics-file",
            str(metrics_file),
            "--decisions-file",
            str(decisions_file),
            "--summary-file",
            str(summary1),
            "--n-jobs",
            "1",
        ]
    )
    main(
        [
            "--trades-file",
            str(trades_file),
            "--metrics-file",
            str(metrics_file),
            "--decisions-file",
            str(decisions_file),
            "--summary-file",
            str(summary2),
            "--n-jobs",
            "2",
        ]
    )

    data1 = json.loads(summary1.read_text())
    data2 = json.loads(summary2.read_text())
    assert data1 == data2
