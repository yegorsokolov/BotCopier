import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_symbol_graph import build_graph
from scripts.generate_mql4_from_model import generate
from scripts.train_target_clone import train, _load_logs, _extract_features
from tests import HAS_NUMPY
import numpy as np

pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="NumPy is required for training tests")


def _write_log(file: Path) -> None:
    fields = [
        "schema_version",
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
        "sl_hit_dist",
        "tp_hit_dist",
    ]
    rows = [
        [
            "2",
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
            "1",
            "2",
            "",
            "0.1",
            "0.0001",
            "100",
            "0",
            "0",
        ],
        [
            "2",
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
            "2",
            "3",
            "",
            "0.1",
            "0.0002",
            "200",
            "0",
            "0",
        ],
    ]
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(fields)
        writer.writerows(rows)


def test_graph_features(tmp_path: Path) -> None:
    data_dir = tmp_path / "logs"
    out_dir = tmp_path / "out"
    data_dir.mkdir()

    log_file = data_dir / "trades_corr.csv"
    _write_log(log_file)

    corr_map = {"EURUSD": ["USDCHF"]}
    extra = {"USDCHF": [0.9, 0.8]}

    df, _, _ = _load_logs(data_dir)
    feat_dicts_tmp, *_ = _extract_features(
        df.to_dict("records"),
        corr_map=corr_map,
        extra_price_series=extra,
    )
    feat_df_tmp = pd.DataFrame(feat_dicts_tmp)
    feat_df_tmp["symbol"] = df["symbol"].values
    feat_df_tmp["price"] = df["price"].astype(float).values
    extra_df = pd.DataFrame(
        {
            "symbol": ["USDCHF"] * len(extra["USDCHF"]),
            "price": extra["USDCHF"],
            "corr_USDCHF": [np.nan] * len(extra["USDCHF"]),
        }
    )
    graph_df = pd.concat(
        [feat_df_tmp[["symbol", "price", "corr_USDCHF"]], extra_df], ignore_index=True
    )
    feat_csv = tmp_path / "features.csv"
    graph_df.to_csv(feat_csv, index=False)

    graph_file = tmp_path / "graph.json"
    build_graph(feat_csv, graph_file)
    graph_parquet = tmp_path / "graph.parquet"
    build_graph(feat_csv, graph_parquet)
    assert graph_parquet.exists()

    feat_dicts, *_ = _extract_features(
        df.to_dict("records"),
        corr_map=corr_map,
        extra_price_series=extra,
        symbol_graph=graph_file,
    )
    feat_df = pd.DataFrame(feat_dicts)
    assert "coint_residual_USDCHF" in feat_df.columns

    train(
        data_dir,
        out_dir,
        corr_map=corr_map,
        extra_price_series=extra,
        symbol_graph=graph_file,
    )

    with open(out_dir / "model.json") as f:
        model = json.load(f)
    feats = model.get("feature_names", [])
    assert "graph_degree" in feats
    assert "graph_pagerank" in feats
    assert "corr_EURUSD_USDCHF" in feats
    assert "coint_residual_USDCHF" in feats
    assert model.get("weighted_by_net_profit") is True
    graph = model.get("graph") or json.load(open(graph_file))
    assert graph.get("symbols") == ["EURUSD", "USDCHF"]
    metrics = graph.get("metrics", {})
    assert metrics.get("degree") == [0.5, 0.5]
    assert metrics.get("pagerank") == [0.5, 0.5]
    coint = graph.get("cointegration", {})
    assert coint.get("EURUSD", {}).get("USDCHF") is not None

    generate(out_dir / "model.json", out_dir, symbol_graph=graph_file)
    mq4_files = list(out_dir.glob("Generated_*.mq4"))
    assert mq4_files, "EA file not generated"
    text = mq4_files[0].read_text()
    assert "GraphDegree()" in text
    assert "GraphPagerank()" in text
    assert 'PairCorrelation("EURUSD", "USDCHF")' in text
    assert 'CointegrationResidual("USDCHF")' in text
    assert 'GraphSymbols[] = {"EURUSD", "USDCHF"}' in text
    assert 'GraphDegreeVals[] = {0.5, 0.5}' in text
    assert 'GraphPagerankVals[] = {0.5, 0.5}' in text

