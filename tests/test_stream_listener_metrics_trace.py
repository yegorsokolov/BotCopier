from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import stream_listener as sl


def test_metric_record_has_errors_and_trace(monkeypatch):
    records = []

    def fake_append(path, record):
        records.append(record)

    monkeypatch.setattr(sl, "append_csv", fake_append)

    msg = SimpleNamespace(
        time="t",
        magic=0,
        winRate=0.1,
        avgProfit=0.2,
        tradeCount=1,
        drawdown=0.0,
        sharpe=0.0,
        fileWriteErrors=1,
        socketErrors=2,
        queueBacklog=3,
        bookRefreshSeconds=0,
    )

    sl.process_metric(msg)

    assert records
    rec = records[0]
    assert rec["file_write_errors"] == 1
    assert rec["socket_errors"] == 2
    assert rec["queue_backlog"] == 3
    assert rec["trace_id"]
    assert rec["span_id"]
