from types import SimpleNamespace

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import stream_listener as sl


def test_capture_system_metrics_includes_trace(monkeypatch):
    sl.current_trace_id = "trace123"
    sl.current_span_id = "span456"

    monkeypatch.setattr(sl.psutil, "cpu_percent", lambda interval=None: 10.0)
    monkeypatch.setattr(sl.psutil, "virtual_memory", lambda: SimpleNamespace(percent=20.0))
    monkeypatch.setattr(
        sl.psutil,
        "net_io_counters",
        lambda: SimpleNamespace(bytes_sent=1, bytes_recv=2),
    )

    records = []

    def fake_append(path, record):
        records.append(record)

    monkeypatch.setattr(sl, "append_csv", fake_append)

    sl.capture_system_metrics()

    assert records
    rec = records[0]
    assert rec["trace_id"] == "trace123"
    assert rec["span_id"] == "span456"
    assert rec["cpu_percent"] == 10.0
    assert rec["mem_percent"] == 20.0
    assert rec["bytes_sent"] == 1
    assert rec["bytes_recv"] == 2
