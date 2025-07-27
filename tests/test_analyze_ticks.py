from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_ticks import load_ticks, compute_metrics, compute_volatility
import pytest


def _write_ticks(file: Path):
    with open(file, "w") as f:
        f.write("time;bid;ask;last;volume\n")
        f.write("2024.01.01 00:00:00;1.1000;1.1002;0;0\n")
        f.write("2024.01.01 00:00:01;1.1001;1.1003;0;0\n")
        f.write("2024.01.01 00:00:02;1.1002;1.1004;0;0\n")


def _write_vol_ticks(file: Path):
    with open(file, "w") as f:
        f.write("time;bid;ask;last;volume\n")
        f.write("2024.01.01 00:00:00;1.1000;1.1002;0;0\n")
        f.write("2024.01.01 00:00:01;1.1002;1.1004;0;0\n")
        f.write("2024.01.01 00:00:02;1.1003;1.1005;0;0\n")
        f.write("2024.01.01 01:00:00;1.1010;1.1012;0;0\n")
        f.write("2024.01.01 01:00:01;1.1015;1.1017;0;0\n")
        f.write("2024.01.01 01:00:02;1.1017;1.1019;0;0\n")


def test_metrics(tmp_path: Path):
    tick_file = tmp_path / "ticks.csv"
    _write_ticks(tick_file)

    rows = load_ticks(tick_file)
    stats = compute_metrics(rows)

    assert stats["tick_count"] == 3
    assert stats["avg_spread"] == pytest.approx(0.0002)
    assert stats["price_change"] == pytest.approx(0.0002)


def test_compute_volatility(tmp_path: Path):
    tick_file = tmp_path / "ticks.csv"
    _write_vol_ticks(tick_file)

    rows = load_ticks(tick_file)
    vols = compute_volatility(rows, interval="hourly")

    assert "2024-01-01 00" in vols
    assert "2024-01-01 01" in vols
    assert vols["2024-01-01 00"] == pytest.approx(5e-05)
