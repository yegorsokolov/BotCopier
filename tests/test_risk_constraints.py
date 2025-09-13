import json
from pathlib import Path

import numpy as np

from botcopier.training.pipeline import train


def test_risk_constraints_penalise_high_drawdown(tmp_path, monkeypatch):
    data = tmp_path / "trades_raw.csv"
    rows = [
        "profit,price,hour,symbol,event_time\n",
        "1,1.0,0,EURUSD,2024-01-01T00:00:00\n",
        "1,1.1,1,EURUSD,2024-01-01T01:00:00\n",
        "-10,1.2,2,EURUSD,2024-01-01T02:00:00\n",
        "1,1.3,3,EURUSD,2024-01-01T03:00:00\n",
        "1,1.4,4,EURUSD,2024-01-01T04:00:00\n",
    ]
    data.write_text("".join(rows))

    def builder(X, y, *, take_bad_trade=True, sample_weight=None, **kwargs):
        meta = {"take_bad_trade": take_bad_trade}
        def predict_fn(arr):
            probs = np.full(arr.shape[0], 0.6)
            if not take_bad_trade and arr.shape[0] > 2:
                probs[2] = 0.0
            return probs
        return meta, predict_fn

    monkeypatch.setattr(
        "botcopier.models.registry.get_model", lambda name: builder
    )
    monkeypatch.setattr(
        "botcopier.training.pipeline.get_model", lambda name: builder
    )
    monkeypatch.setattr(
        "botcopier.training.pipeline._classification_metrics",
        lambda y, p, r, selected=None: {"roc_auc": 1.0, "accuracy": 1.0},
    )
    monkeypatch.setattr("botcopier.training.pipeline.linkage", lambda *a, **k: np.zeros((1, 4)))
    monkeypatch.setattr("botcopier.training.pipeline.fcluster", lambda *a, **k: np.array([1, 1]))
    monkeypatch.setattr("botcopier.training.pipeline.np.cov", lambda *a, **k: np.eye(2))
    monkeypatch.setattr(
        "botcopier.training.pipeline.PurgedWalkForward.split",
        lambda self, X: [(np.arange(len(X)), np.arange(len(X)))],
    )

    out_dir = tmp_path / "out"
    param_grid = [{"take_bad_trade": True}, {"take_bad_trade": False}]
    train(
        data,
        out_dir,
        param_grid=param_grid,
        max_drawdown=5.0,
        var_limit=5.0,
    )
    model = json.loads((out_dir / "model.json").read_text())
    assert model["take_bad_trade"] is False
    assert model["risk_params"] == {"max_drawdown": 5.0, "var_limit": 5.0}
    assert model["risk_metrics"]["max_drawdown"] <= 5.0
