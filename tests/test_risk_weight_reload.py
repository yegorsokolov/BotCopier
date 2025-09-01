import json
import time
from pathlib import Path


class _WeightLogger:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.weights: dict[str, float] = {}
        self.ts = 0.0
        self.logs: list[float] = []

    def _load(self) -> None:
        data = json.loads(self.model_path.read_text())
        syms = data.get("risk_parity_symbols", [])
        wts = data.get("risk_parity_weights", [])
        self.weights = {s: float(w) for s, w in zip(syms, wts)}

    def on_init(self) -> None:
        self._load()
        self.ts = self.model_path.stat().st_mtime

    def on_timer(self) -> None:
        ts = self.model_path.stat().st_mtime
        if ts != self.ts:
            self._load()
            self.ts = ts

    def log_weight(self, symbol: str) -> float:
        w = self.weights.get(symbol, 1.0)
        self.logs.append(w)
        return w


def test_weight_reload(tmp_path: Path) -> None:
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps({"risk_parity_symbols": ["EURUSD"], "risk_parity_weights": [1.0]})
    )
    logger = _WeightLogger(model)
    logger.on_init()
    first = logger.log_weight("EURUSD")

    time.sleep(1)
    model.write_text(
        json.dumps({"risk_parity_symbols": ["EURUSD"], "risk_parity_weights": [0.5]})
    )
    logger.on_timer()
    second = logger.log_weight("EURUSD")

    assert first != second
    assert logger.logs == [1.0, 0.5]

