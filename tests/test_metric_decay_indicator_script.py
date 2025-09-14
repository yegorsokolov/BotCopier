import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import metrics_collector as mc


class DummyDetector:
    def __init__(self):
        self.estimation = 1.0
        self.calls = 0

    def update(self, x: float) -> bool:
        self.calls += 1
        changed = self.calls > 1
        self.estimation = x
        return changed


def test_metric_decay_triggers_script(monkeypatch, tmp_path):
    called = []
    script = tmp_path / "sym_ind.py"
    script.write_text("#")

    def fake_run(cmd, check):
        called.append(cmd)
        model = Path(cmd[-1])
        model.write_text(
            json.dumps(
                {
                    "symbolic_indicators": {
                        "feature_names": ["win_rate", "avg_profit"],
                        "formulas": ["add(win_rate,avg_profit)"],
                    }
                }
            )
        )

    monkeypatch.setattr(mc.subprocess, "run", fake_run)
    monkeypatch.setattr(mc.technical, "refresh_symbolic_indicators", lambda _m: None)

    async def _run():
        db = tmp_path / "m.db"
        model = tmp_path / "model.json"
        q = asyncio.Queue()
        base_row = {
            "time": "t",
            "magic": "0",
            "win_rate": "1.0",
            "avg_profit": "0.2",
            "trade_count": "1",
            "drawdown": "0.0",
            "sharpe": "0.0",
            "sortino": "0.0",
            "expectancy": "0.0",
            "cvar": "0.0",
            "roc_auc": "0.0",
            "pr_auc": "0.0",
            "brier_score": "0.0",
            "file_write_errors": 0,
            "socket_errors": 0,
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
            "trace_id": "trace",
            "span_id": "span",
            "queue_backlog": 0,
        }
        await q.put(base_row)
        low_row = dict(base_row)
        low_row["win_rate"] = "0.0"
        await q.put(low_row)
        await q.put(None)
        await mc._writer_task(
            db,
            q,
            lambda _r: None,
            drift_metric="win_rate",
            drift_threshold=0.5,
            model_json=model,
            detector=DummyDetector(),
            indicator_script=script,
        )
        data = json.loads(model.read_text())
        assert data["symbolic_indicators"]["formulas"] == ["add(win_rate,avg_profit)"]
        df = pd.DataFrame({"win_rate": [1.0], "avg_profit": [0.2]})
        df2, feats = mc.technical._apply_symbolic_indicators(
            df, ["win_rate", "avg_profit"], model
        )
        assert "sym_0" in df2.columns
        assert "sym_0" in feats

    asyncio.run(_run())
    assert called and str(script) in called[0]
