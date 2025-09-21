"""Tests for :func:`botcopier.training.preprocessing.should_use_lightweight`."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "botcopier" / "training" / "preprocessing.py"
SPEC = importlib.util.spec_from_file_location("_preprocessing_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
preprocessing = importlib.util.module_from_spec(SPEC)
sys.modules["_preprocessing_test_module"] = preprocessing
SPEC.loader.exec_module(preprocessing)

should_use_lightweight = preprocessing.should_use_lightweight
LIGHTWEIGHT_ROW_THRESHOLD = preprocessing.LIGHTWEIGHT_ROW_THRESHOLD


def _write_csv(path: Path, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def test_should_use_lightweight_with_header(tmp_path: Path) -> None:
    data_dir = tmp_path
    csv = data_dir / "trades_raw.csv"
    rows = [
        "time;bid;ask;f0;latency",
        "2024.01.01 00:00:00;1.0;1.1;0.2;10",
        "2024.01.01 00:00:01;1.3;1.4;-0.2;15",
    ]
    _write_csv(csv, rows)
    assert should_use_lightweight(data_dir, {}) is True


def test_should_use_lightweight_without_header(tmp_path: Path) -> None:
    data_dir = tmp_path
    csv = data_dir / "trades_raw.csv"
    rows = [
        "2024.01.01 00:00:00;1.0;1.1;0.2;10",
        "2024.01.01 00:00:01;1.3;1.4;-0.2;15",
    ]
    _write_csv(csv, rows)
    assert should_use_lightweight(data_dir, {}) is True


def test_should_use_lightweight_exits_early(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data_dir = tmp_path
    csv = data_dir / "trades_raw.csv"
    csv.write_text("placeholder", encoding="utf-8")

    header = "time;bid;ask\n"
    body = "".join(
        f"2024.01.01 00:00:{idx:02d};1.0;1.1\n" for idx in range(LIGHTWEIGHT_ROW_THRESHOLD + 50)
    )
    content = header + body

    class GuardedStringIO(io.StringIO):
        def __init__(self, text: str, max_calls: int) -> None:
            super().__init__(text)
            self.max_calls = max_calls
            self.calls = 0

        def readline(self, *args, **kwargs):  # type: ignore[override]
            self.calls += 1
            if self.calls > self.max_calls:
                raise AssertionError("readline called more often than expected")
            return super().readline(*args, **kwargs)

    opened: list[GuardedStringIO] = []
    original_open = Path.open

    def fake_open(self: Path, *args, **kwargs):  # type: ignore[override]
        if self == csv:
            handle = GuardedStringIO(content, LIGHTWEIGHT_ROW_THRESHOLD + 5)
            opened.append(handle)
            return handle
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)

    assert should_use_lightweight(data_dir, {}) is False
    assert opened, "expected patched path to be opened"
    assert opened[0].calls <= LIGHTWEIGHT_ROW_THRESHOLD + 2
