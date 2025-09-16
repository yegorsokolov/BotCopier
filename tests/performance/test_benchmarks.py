from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pytest_benchmark")

from botcopier.data.loading import _load_logs
from botcopier.features.engineering import _extract_features
from botcopier.scripts.evaluation import evaluate


def test_load_logs_benchmark(benchmark):
    """Benchmark the log loading routine."""
    data_file = Path("tests/fixtures/trades_small.csv")
    benchmark(lambda: _load_logs(data_file))


def test_feature_engineering_benchmark(benchmark):
    """Benchmark feature extraction."""
    data_file = Path("tests/fixtures/trades_small.csv")
    df, feature_cols, _ = _load_logs(data_file)
    benchmark(lambda: _extract_features(df.copy(), feature_names=list(feature_cols)))


def test_evaluation_benchmark(benchmark, tmp_path):
    """Benchmark evaluation pipeline on small synthetic data."""

    preds = pd.DataFrame(
        {
            "timestamp": ["2024.01.01 00:00:00", "2024.01.01 00:02:00"],
            "symbol": ["EURUSD", "EURUSD"],
            "direction": [1, -1],
            "lots": [0.1, 0.2],
            "probability": [0.7, 0.4],
            "value": [1.2, -0.5],
            "log_variance": [np.log(0.1), np.log(0.2)],
            "executed_model_idx": [0, 1],
            "decision_id": [1, 2],
        }
    )
    trades = pd.DataFrame(
        {
            "event_time": ["2024.01.01 00:00:10", "2024.01.01 00:02:30"],
            "action": ["OPEN", "OPEN"],
            "order_type": ["0", "1"],
            "symbol": ["EURUSD", "EURUSD"],
            "lots": [0.1, 0.2],
            "ticket": ["1", "2"],
            "executed_model_idx": [0, 1],
            "decision_id": [1, 2],
            "profit": [1.0, -0.6],
        }
    )
    pred_file = tmp_path / "preds.csv"
    trades_file = tmp_path / "trades.csv"
    preds.to_csv(pred_file, sep=";", index=False)
    trades.to_csv(trades_file, sep=";", index=False)

    benchmark(lambda: evaluate(pred_file, trades_file, window=60))
