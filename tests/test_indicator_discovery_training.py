import asyncio
import json

from botcopier.training.pipeline import train as train_pipeline
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


def test_discovered_indicators_used_in_training(monkeypatch, tmp_path):
    def fake_evolve(df, cols, model_path):
        formula = "add(win_rate,avg_profit)"
        model_path.write_text(
            json.dumps(
                {
                    "symbolic_indicators": {
                        "feature_names": list(cols),
                        "formulas": [formula],
                    }
                }
            )
        )
        return [formula]

    monkeypatch.setattr(mc.indicator_discovery, "evolve_indicators", fake_evolve)
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
        )
        data = json.loads(model.read_text())
        assert data["symbolic_indicators"]["formulas"] == ["add(win_rate,avg_profit)"]

        train_csv = tmp_path / "trades_raw.csv"
        rows = ["label,win_rate,avg_profit\n"]
        for i in range(20):
            wr = 0.4 + 0.02 * i
            ap = 0.1 + 0.01 * i
            label = 1 if wr + ap > 0.7 else 0
            rows.append(f"{label},{wr},{ap}\n")
        train_csv.write_text("".join(rows))
        out_dir = tmp_path / "out"
        train_pipeline(train_csv, out_dir, model_json=model, cluster_correlation=1.0)
        model_trained = json.loads((out_dir / "model.json").read_text())
        assert any(f.startswith("sym_") for f in model_trained["feature_names"])
        assert model_trained["symbolic_indicators"]["formulas"] == [
            "add(win_rate,avg_profit)"
        ]

    asyncio.run(_run())
