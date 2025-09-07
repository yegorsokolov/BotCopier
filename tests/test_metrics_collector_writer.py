import asyncio
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import metrics_collector as mc


def test_writer_persists_trace_and_errors(tmp_path):
    async def _run():
        db = tmp_path / "m.db"
        q = asyncio.Queue()
        row = {
        "time": "t",
        "magic": "0",
        "win_rate": "0.1",
        "avg_profit": "0.2",
        "trade_count": "1",
        "drawdown": "0.0",
        "sharpe": "0.0",
        "sortino": "0.0",
        "expectancy": "0.0",
        "cvar": "0.0",
        "file_write_errors": 1,
        "socket_errors": 2,
        "cpu_load": "0",
        "flush_latency_ms": "0",
        "network_latency_ms": "0",
        "book_refresh_seconds": "0",
        "var_breach_count": "0",
        "trade_queue_depth": "0",
        "metric_queue_depth": "0",
        "trade_retry_count": "0",
        "metric_retry_count": "0",
        "fallback_events": 0,
        "risk_weight": "0",
        "trace_id": "trace123",
        "span_id": "span456",
        "queue_backlog": 5,
        }
        await q.put(row)
        await q.put(None)
        await mc._writer_task(db, q, lambda _r: None)
        conn = sqlite3.connect(db)
        try:
            cur = conn.execute(
                "SELECT file_write_errors, socket_errors, queue_backlog, trace_id, span_id FROM metrics"
            )
            data = cur.fetchone()
        finally:
            conn.close()
        assert data == ("1", "2", "5", "trace123", "span456")

    asyncio.run(_run())
