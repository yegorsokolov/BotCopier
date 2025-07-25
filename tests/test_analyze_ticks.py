from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_ticks import load_ticks, compute_metrics
import pytest


def _write_ticks(file: Path):
    with open(file, "w") as f:
        f.write("time;bid;ask;last;volume\n")
        f.write("2024.01.01 00:00:00;1.1000;1.1002;0;0\n")
        f.write("2024.01.01 00:00:01;1.1001;1.1003;0;0\n")
        f.write("2024.01.01 00:00:02;1.1002;1.1004;0;0\n")


def test_metrics(tmp_path: Path):
    tick_file = tmp_path / "ticks.csv"
    _write_ticks(tick_file)

    rows = load_ticks(tick_file)
    stats = compute_metrics(rows)

    assert stats["tick_count"] == 3
    assert stats["avg_spread"] == pytest.approx(0.0002)
    assert stats["price_change"] == pytest.approx(0.0002)
