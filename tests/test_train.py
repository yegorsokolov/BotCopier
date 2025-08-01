import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests import HAS_NUMPY, HAS_TF
from scripts.train_target_clone import train, _load_logs, _load_calendar, _extract_features

pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="NumPy is required for training tests")


def _write_log(file: Path):
    fields = [
        "event_id",
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "profit",
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
    ]
    rows = [
        [
            "1",
            "2024.01.01 00:00:00",
            "",
            "",
            "OPEN",
            "1",
            "",
            "",
            "EURUSD",
            "0",
            "0.1",
            "1.1000",
            "1.0950",
            "1.1100",
            "0",
            "2",
            "",
            "0.1",
            "0.0001",
            "100",
        ],
        [
            "2",
            "2024.01.01 01:00:00",
            "",
            "",
            "OPEN",
            "2",
            "",
            "",
            "EURUSD",
            "1",
            "0.1",
            "1.2000",
            "1.1950",
            "1.2100",
            "0",
            "3",
            "",
            "0.1",
            "0.0002",
            "200",
        ],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def _write_log_many(file: Path, count: int = 10):
    fields = [
        "event_id",
        "event_time",
        "broker_time",
        "local_time",
        "action",
        "ticket",
        "magic",
        "source",
        "symbol",
        "order_type",
        "lots",
        "price",
        "sl",
        "tp",
        "profit",
        "spread",
        "comment",
        "remaining_lots",
        "slippage",
        "volume",
    ]
    rows = []
    for i in range(count):
        hour = i % 24
        order_type = "0" if i % 2 == 0 else "1"
        rows.append([
            str(i + 1),
            f"2024.01.01 {hour:02d}:00:00",
            "",
            "",
            "OPEN",
            str(i + 1),
            "",
            "",
            "EURUSD",
            order_type,
            "0.1",
            "1.1000",
            "1.0950",
            "1.1100",
            "0",
            "2",
            "",
            "0.1",
            "0.0001",
            str(100 + i),
        ])
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def _write_metrics(file: Path):
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["time", "magic", "win_rate", "avg_profit", "trade_count"])
        writer.writerow([
            "2024.01.02 00:00",
            "0",
            "0.5",
            "1.0",
            "2",
        ])


def test_train(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert "coefficients" in data
    assert "threshold" in data
    assert "day_of_week" not in data.get("feature_names", [])
    assert "hour" not in data.get("feature_names", [])
    assert "hour_sin" in data.get("feature_names", [])
    assert "hour_cos" in data.get("feature_names", [])
    assert "dow_sin" in data.get("feature_names", [])
    assert "dow_cos" in data.get("feature_names", [])
    assert "spread" in data.get("feature_names", [])
    assert "equity" in data.get("feature_names", [])
    assert "margin_level" in data.get("feature_names", [])
    assert data.get("weighted") is True
    assert "feature_mean" in data
    assert "feature_std" in data

    init_file = out_dir / "policy_init.json"
    assert init_file.exists()
    with open(init_file) as f:
        init = json.load(f)
    assert "weights" in init
    assert "intercepts" in init


def test_train_with_indicators(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, use_sma=True, use_rsi=True, use_macd=True)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert any(name in data.get("feature_names", []) for name in ["sma", "rsi", "macd"])


def test_train_with_atr_bollinger(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, use_atr=True, use_bollinger=True)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    feats = data.get("feature_names", [])
    assert "atr" in feats
    assert any(n.startswith("bollinger_") for n in feats)


def test_train_with_stochastic_adx(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, use_stochastic=True, use_adx=True)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    feats = data.get("feature_names", [])
    assert "stochastic_k" in feats
    assert "stochastic_d" in feats
    assert "adx" in feats


def test_train_with_volatility(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    vols = {"2024-01-01 00": 0.1, "2024-01-01 01": 0.2}
    train(data_dir, out_dir, volatility_series=vols)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert "volatility" in data.get("feature_names", [])


def test_train_with_calendar(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    cal_file = Path(__file__).with_name("sample_calendar.csv")
    events = _load_calendar(cal_file)

    train(data_dir, out_dir, calendar_events=events, event_window=60)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    feats = data.get("feature_names", [])
    assert "event_flag" in feats
    assert "event_impact" in feats


def test_load_logs_with_metrics(tmp_path: Path):
    data_dir = tmp_path / "logs"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)
    metrics_file = data_dir / "metrics.csv"
    _write_metrics(metrics_file)

    df = _load_logs(data_dir)
    assert "win_rate" in df.columns
    assert "spread" in df.columns
    assert "slippage" in df.columns


def test_train_xgboost(tmp_path: Path):
    pytest.importorskip("xgboost")
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, model_type="xgboost", n_estimators=10)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("model_type") == "xgboost"
    assert "coefficients" in data
    assert len(data.get("probability_table", [])) == 24
    assert data.get("weighted") is False


def test_train_lightgbm(tmp_path: Path):
    pytest.importorskip("lightgbm")
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, model_type="lgbm", n_estimators=10)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("model_type") == "lgbm"
    assert "coefficients" in data
    assert len(data.get("probability_table", [])) == 24
    assert data.get("weighted") is False


def test_train_catboost(tmp_path: Path):
    pytest.importorskip("catboost")
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, model_type="catboost", n_estimators=10)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("model_type") == "catboost"
    assert "coefficients" in data
    assert len(data.get("probability_table", [])) == 24
    assert data.get("weighted") is False


def test_train_nn(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, model_type="nn", early_stop=True)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("model_type") == "nn"
    assert "nn_weights" in data
    assert data.get("hidden_size", 0) > 0
    assert data.get("weighted") is False


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_train_lstm(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, model_type="lstm", sequence_length=3, early_stop=True)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("model_type") == "lstm"
    assert "lstm_weights" in data
    assert data.get("sequence_length") == 3
    assert data.get("weighted") is False


@pytest.mark.skipif(not HAS_TF, reason="TensorFlow required")
def test_train_transformer(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_test.csv"
    _write_log(log_file)

    train(data_dir, out_dir, model_type="transformer", sequence_length=3, early_stop=True)

    model_file = out_dir / "model.json"
    assert model_file.exists()
    with open(model_file) as f:
        data = json.load(f)
    assert data.get("model_type") == "transformer"
    assert "transformer_weights" in data
    assert data.get("sequence_length") == 3
    assert data.get("weighted") is False


def test_incremental_train(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()

    log_file1 = data_dir / "trades_a.csv"
    _write_log(log_file1)

    train(data_dir, out_dir)

    log_file2 = data_dir / "trades_b.csv"
    _write_log(log_file2)

    train(data_dir, out_dir, incremental=True)

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    assert data.get("num_samples", 0) >= 4


def test_feature_cache(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()

    log_file = data_dir / "trades_cache.csv"
    _write_log(log_file)

    train(data_dir, out_dir, cache_features=True)

    cache_file = out_dir / "feature_cache.npz"
    assert cache_file.exists()

    # remove logs to ensure training loads from cache
    for f in data_dir.glob("*"):
        f.unlink()

    train(data_dir, out_dir, incremental=True, cache_features=True)

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    assert data.get("num_samples", 0) > 0


def test_hourly_thresholds(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_many.csv"
    _write_log_many(log_file, count=12)

    train(data_dir, out_dir)

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    ht = data.get("hourly_thresholds")
    assert isinstance(ht, list)
    assert len(ht) == 24


def test_corr_features(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_corr.csv"
    _write_log(log_file)
    extra = {"USDCHF": [0.9, 0.8]}

    train(data_dir, out_dir, corr_pairs=[("EURUSD", "USDCHF")], extra_price_series=extra)

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    feats = data.get("feature_names", [])
    assert "ratio_EURUSD_USDCHF" in feats
    assert "corr_EURUSD_USDCHF" in feats

    df = _load_logs(data_dir)
    feature_dicts, *_ = _extract_features(
        df.to_dict("records"),
        corr_pairs=[("EURUSD", "USDCHF")],
        extra_price_series=extra,
    )
    assert feature_dicts[1]["ratio_EURUSD_USDCHF"] == pytest.approx(1.1000 / 0.9)


def test_slippage_feature(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_slip.csv"
    _write_log(log_file)

    train(data_dir, out_dir, use_slippage=True)

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    feats = data.get("feature_names", [])
    assert "slippage" in feats


def test_volume_feature(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_vol.csv"
    _write_log(log_file)

    df = _load_logs(data_dir)
    assert "volume" in df.columns
    assert int(df["volume"].iloc[0]) == 100

    train(data_dir, out_dir, use_volume=True)

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    feats = data.get("feature_names", [])
    assert "volume" in feats


def test_encoder_regime(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_reg.csv"
    _write_log(log_file)

    enc = {"window": 1, "weights": [[1.0]], "centers": [[0.0], [1.0]]}
    enc_file = tmp_path / "enc.json"
    with open(enc_file, "w") as f:
        json.dump(enc, f)

    train(data_dir, out_dir, encoder_file=enc_file)

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    feats = data.get("feature_names", [])
    assert "regime" in feats
    assert "encoder_centers" in data


def test_higher_timeframe_features(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_many.csv"
    _write_log_many(log_file, count=6)

    train(
        data_dir,
        out_dir,
        use_sma=True,
        use_rsi=True,
        use_macd=True,
        higher_timeframes=["H1"],
    )

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    feats = data.get("feature_names", [])
    assert "sma_H1" in feats
    assert "rsi_H1" in feats
    assert "macd_H1" in feats


def test_train_regress_sl_tp(tmp_path: Path):
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    log_file = data_dir / "trades_reg_sl.csv"
    _write_log(log_file)

    train(data_dir, out_dir, regress_sl_tp=True)

    with open(out_dir / "model.json") as f:
        data = json.load(f)
    assert "sl_coefficients" in data
    assert "tp_coefficients" in data
