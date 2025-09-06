import json
import socket
import subprocess
import sys
from pathlib import Path

import grpc
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "proto"))
import log_service_pb2_grpc  # type: ignore
import trade_event_pb2  # type: ignore

from scripts import grpc_log_service
from scripts.train_target_clone import _load_logs, train


def test_generated_features(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "feature_names": [
                    "spread",
                    "slippage",
                    "equity",
                    "margin_level",
                    "volume",
                    "hour_sin",
                    "hour_cos",
                ]
            }
        )
    )

    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text("#property strict\n\n// __GET_FEATURE__\n")

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4_from_model.py",
            "--model",
            model,
            "--template",
            template,
        ],
        check=True,
    )

    content = template.read_text()
    assert "case 0: return MarketInfo(Symbol(), MODE_SPREAD); // spread" in content
    assert "case 1: return OrderSlippage(); // slippage" in content
    assert "case 2: return AccountEquity(); // equity" in content
    assert "case 3: return AccountMarginLevel(); // margin_level" in content
    assert "case 4: return iVolume(Symbol(), PERIOD_CURRENT, 0); // volume" in content
    assert (
        "case 5: return MathSin(TimeHour(TimeCurrent())*2*MathPi()/24); // hour_sin"
        in content
    )
    assert (
        "case 6: return MathCos(TimeHour(TimeCurrent())*2*MathPi()/24); // hour_cos"
        in content
    )

    data = json.loads(model.read_text())
    assert data["feature_names"] == [
        "spread",
        "slippage",
        "equity",
        "margin_level",
        "volume",
        "hour_sin",
        "hour_cos",
    ]


def test_session_models_inserted(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(
        json.dumps(
            {
                "feature_names": [],
                "session_models": {
                    "asian": {
                        "coefficients": [1.0],
                        "intercept": 0.1,
                        "threshold": 0.5,
                        "feature_mean": [0.0],
                        "feature_std": [1.0],
                    }
                },
            }
        )
    )

    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text("#property strict\n\n// __SESSION_MODELS__\n")

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_mql4_from_model.py",
            "--model",
            model,
            "--template",
            template,
        ],
        check=True,
    )

    content = template.read_text()
    assert "g_coeffs_asian" in content
    assert "g_threshold_asian" in content
    assert "g_feature_mean_asian" in content
    assert "g_feature_std_asian" in content

    data = json.loads(model.read_text())
    assert "feature_mean" in data["session_models"]["asian"]
    assert "feature_std" in data["session_models"]["asian"]


def test_generation_fails_on_unmapped_feature(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(json.dumps({"feature_names": ["unknown"]}))

    template = tmp_path / "StrategyTemplate.mq4"
    template.write_text("#property strict\n\n// __GET_FEATURE__\n")

    with pytest.raises(subprocess.CalledProcessError) as exc:
        subprocess.run(
            [
                sys.executable,
                "scripts/generate_mql4_from_model.py",
                "--model",
                model,
                "--template",
                template,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    assert "Update StrategyTemplate.mq4" in exc.value.stderr


def test_scaler_stats_present(tmp_path):
    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,spread,hour\n" "0,1.0,1\n" "1,1.2,2\n" "0,1.3,9\n" "1,1.5,10\n" "0,1.4,17\n" "1,1.6,18\n"
    )
    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    for sess in ["asian", "london", "newyork"]:
        params = model["session_models"][sess]
        assert "feature_mean" in params
        assert "feature_std" in params


def test_threshold_and_metrics_present(tmp_path):
    data = tmp_path / "trades_raw.csv"
    rows = ["label,spread,hour\n"]
    for base in [1, 9, 17]:
        for i in range(8):
            label = i % 2
            spread = 1.0 + 0.1 * i + (0.5 if label else 0)
            rows.append(f"{label},{spread},{base + i}\n")
        rows.append(f"0,1.0,{base + 8}\n")
        rows.append(f"1,2.0,{base + 9}\n")
    data.write_text("".join(rows))

    out_dir = tmp_path / "out"
    train(data, out_dir)
    model = json.loads((out_dir / "model.json").read_text())
    for sess in ["asian", "london", "newyork"]:
        params = model["session_models"][sess]
        assert "threshold" in params
        assert "metrics" in params
        assert "accuracy" in params["metrics"]
        assert "recall" in params["metrics"]


def test_log_trade_captures_extra_fields(tmp_path):
    host = "127.0.0.1"
    srv_sock = socket.socket()
    srv_sock.bind((host, 0))
    port = srv_sock.getsockname()[1]
    srv_sock.close()

    trade_out = tmp_path / "trades.csv"
    metrics_out = tmp_path / "metrics.csv"
    server = grpc_log_service.create_server(host, port, trade_out, metrics_out)
    server.start()
    try:
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = log_service_pb2_grpc.LogServiceStub(channel)
        trade = trade_event_pb2.TradeEvent(
            event_id=1,
            event_time="t",
            broker_time="b",
            local_time="l",
            action="OPEN",
            ticket=1,
            magic=2,
            source="mt4",
            symbol="EURUSD",
            order_type=0,
            lots=0.1,
            price=1.2345,
            slippage=0.5,
            equity=1000.0,
            margin_level=200.0,
        )
        stub.LogTrade(trade)
    finally:
        server.stop(0)

    text = trade_out.read_text()
    assert "0.5" in text
    assert "1000.0" in text
    assert "200.0" in text


def test_load_logs_optional_features(tmp_path):
    csv = tmp_path / "trades_raw.csv"
    csv.write_text(
        "label,spread,slippage,equity,margin_level,volume,hour\n"
        "0,1.0,0.5,1000,200,100,12\n"
    )

    df, feature_cols, _ = _load_logs(tmp_path)
    assert set(feature_cols) == {
        "spread",
        "slippage",
        "equity",
        "margin_level",
        "volume",
        "hour_sin",
        "hour_cos",
    }
    for col in ["slippage", "equity", "margin_level"]:
        assert col in df.columns

