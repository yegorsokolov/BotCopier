import json
from pathlib import Path

import numpy as np
import pandas as pd

from botcopier.data.loading import _load_logs
from botcopier.training.pipeline import train


def test_meta_labels_not_null(tmp_path: Path) -> None:
    data = tmp_path / "trades_raw.csv"
    rows = [
        "label,price,spread,hour\n",
        "1,1.0,0.05,0\n",
        "0,0.9,0.05,1\n",
        "1,1.1,0.05,2\n",
    ]
    data.write_text("".join(rows))
    df, _, _ = _load_logs(
        data,
        take_profit_mult=1.0,
        stop_loss_mult=1.0,
        hold_period=2,
    )
    assert df["meta_label"].notna().all()


def test_meta_labeling_changes_labels(tmp_path: Path) -> None:
    rows = [
        [0, 1.00, 0.05, 0],
        [1, 0.89, 0.05, 1],
        [0, 1.05, 0.05, 2],
        [1, 1.30, 0.05, 3],
        [0, 1.10, 0.05, 4],
        [1, 0.95, 0.05, 5],
        [0, 0.96, 0.05, 6],
        [1, 0.97, 0.05, 7],
        [0, 1.20, 0.05, 8],
        [1, 1.00, 0.05, 9],
    ]
    raw = pd.DataFrame(rows, columns=["label", "price", "spread", "hour"])
    raw_file = tmp_path / "raw.csv"
    raw.to_csv(raw_file, index=False)
    loaded, _, _ = _load_logs(
        raw_file,
        take_profit_mult=1.0,
        stop_loss_mult=1.0,
        hold_period=2,
    )
    # Meta labels should differ from original labels for at least one row
    assert (loaded["label"] != loaded["meta_label"]).any()


def _legacy_meta(prices: np.ndarray, tp: np.ndarray, sl: np.ndarray, hold_period: int):
    n = len(prices)
    horizon_idx = (np.arange(n) + hold_period).clip(0, n - 1)
    horizon_len = horizon_idx - np.arange(n)
    tp_time = np.full(n, hold_period + 1, dtype=int)
    sl_time = np.full(n, hold_period + 1, dtype=int)
    meta = np.zeros(n, dtype=float)
    for i in range(n):
        end = horizon_idx[i]
        h = horizon_len[i]
        for j in range(i + 1, end + 1):
            p = prices[j]
            offset = j - i
            if tp_time[i] > h and p >= tp[i]:
                tp_time[i] = offset
            if sl_time[i] > h and p <= sl[i]:
                sl_time[i] = offset
            if tp_time[i] <= h and sl_time[i] <= h:
                break
        if tp_time[i] <= sl_time[i] and tp_time[i] <= h:
            meta[i] = 1.0
        if tp_time[i] > h:
            tp_time[i] = h + 1
        if sl_time[i] > h:
            sl_time[i] = h + 1
    return horizon_idx, tp_time, sl_time, meta


def test_vectorized_meta_matches_legacy(tmp_path: Path) -> None:
    rows = [
        [1, 1.00, 0.05, 0],
        [1, 0.89, 0.05, 1],
        [1, 1.05, 0.05, 2],
        [1, 1.30, 0.05, 3],
    ]
    raw = pd.DataFrame(rows, columns=["label", "price", "spread", "hour"])
    raw_file = tmp_path / "raw.csv"
    raw.to_csv(raw_file, index=False)
    loaded, _, _ = _load_logs(
        raw_file, take_profit_mult=1.0, stop_loss_mult=1.0, hold_period=2
    )
    prices = raw["price"].to_numpy()
    spreads = raw["spread"].to_numpy()
    tp = prices + spreads
    sl = prices - spreads
    expected_h, expected_tp, expected_sl, expected_meta = _legacy_meta(
        prices, tp, sl, 2
    )
    assert np.array_equal(loaded["horizon"].to_numpy(), expected_h)
    assert np.array_equal(loaded["tp_time"].to_numpy(), expected_tp)
    assert np.array_equal(loaded["sl_time"].to_numpy(), expected_sl)
    assert np.array_equal(loaded["meta_label"].to_numpy(), expected_meta)
