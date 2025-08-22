import logging
from types import SimpleNamespace
from pathlib import Path
import sys
import types

pa_stub = types.SimpleNamespace(
    int32=lambda *a, **k: None,
    string=lambda *a, **k: None,
    float64=lambda *a, **k: None,
    schema=lambda *a, **k: None,
)
sys.modules.setdefault("pyarrow", pa_stub)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import stream_listener as sl


def test_trade_validation(monkeypatch, caplog):
    records = []

    def fake_append(path, record):
        records.append(record)

    monkeypatch.setattr(sl, "append_csv", fake_append)

    msg = SimpleNamespace(
        eventId="bad",  # invalid type
        eventTime="t",
        brokerTime="b",
        localTime="l",
        action="OPEN",
        ticket=1,
        magic=0,
        source="src",
        symbol="X",
        orderType=0,
        lots=0.1,
        price=1.0,
        sl=0.0,
        tp=0.0,
        profit=0.0,
        comment="",
        remainingLots=0.0,
        decisionId=0,
    )
    with caplog.at_level(logging.WARNING):
        sl.process_trade(msg)
    assert not records
    assert "invalid trade event" in caplog.text


def test_metric_validation(monkeypatch, caplog):
    records = []

    def fake_append(path, record):
        records.append(record)

    monkeypatch.setattr(sl, "append_csv", fake_append)

    msg = SimpleNamespace(
        time="t",
        magic=0,
        winRate=0.1,
        avgProfit=0.2,
        tradeCount="bad",  # invalid type
        drawdown=0.0,
        sharpe=0.0,
        fileWriteErrors=0,
        socketErrors=0,
        bookRefreshSeconds=0,
    )
    with caplog.at_level(logging.WARNING):
        sl.process_metric(msg)
    assert not records
    assert "invalid metric event" in caplog.text


def test_trade_validation_missing_field(monkeypatch, caplog):
    records = []

    def fake_append(path, record):
        records.append(record)

    monkeypatch.setattr(sl, "append_csv", fake_append)

    msg = SimpleNamespace(
        eventId=1,
        eventTime="t",
        brokerTime="b",
        localTime="l",
        ticket=1,
        magic=0,
        source="src",
        symbol="X",
        orderType=0,
        lots=0.1,
        price=1.0,
        sl=0.0,
        tp=0.0,
        profit=0.0,
        comment="",
        remainingLots=0.0,
        decisionId=0,
    )  # action missing
    with caplog.at_level(logging.WARNING):
        sl.process_trade(msg)
    assert not records
    assert "invalid trade event" in caplog.text


def test_trade_schema_version_mismatch(monkeypatch, caplog):
    records = []

    def fake_append(path, record):
        records.append(record)

    monkeypatch.setattr(sl, "append_csv", fake_append)

    msg = SimpleNamespace(
        schemaVersion=sl.SCHEMA_VERSION + 1,
        eventId=1,
        eventTime="t",
        brokerTime="b",
        localTime="l",
        action="OPEN",
        ticket=1,
        magic=0,
        source="src",
        symbol="X",
        orderType=0,
        lots=0.1,
        price=1.0,
        sl=0.0,
        tp=0.0,
        profit=0.0,
        comment="",
        remainingLots=0.0,
        decisionId=0,
    )
    with caplog.at_level(logging.WARNING):
        sl.process_trade(msg)
    assert not records
    assert "schema version mismatch" in caplog.text


def test_metric_schema_version_mismatch(monkeypatch, caplog):
    records = []

    def fake_append(path, record):
        records.append(record)

    monkeypatch.setattr(sl, "append_csv", fake_append)

    msg = SimpleNamespace(
        schemaVersion=sl.SCHEMA_VERSION + 1,
        time="t",
        magic=0,
        winRate=0.1,
        avgProfit=0.2,
        tradeCount=1,
        drawdown=0.0,
        sharpe=0.0,
        fileWriteErrors=0,
        socketErrors=0,
        bookRefreshSeconds=0,
    )
    with caplog.at_level(logging.WARNING):
        sl.process_metric(msg)
    assert not records
    assert "schema version mismatch" in caplog.text
