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
            "1",
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
            "0",
            "0",
        ],
        [
            "1",
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
    feat_dicts, *_ = _extract_features(
        df.to_dict("records"),
        corr_map=corr_map,
        extra_price_series=extra,
    )
    feat_df = pd.DataFrame(feat_dicts)
    feat_csv = tmp_path / "features.csv"
    feat_df.to_csv(feat_csv, index=False)

    graph_file = tmp_path / "graph.json"
    build_graph(feat_csv, graph_file)
    graph_parquet = tmp_path / "graph.parquet"
    build_graph(feat_csv, graph_parquet)
    assert graph_parquet.exists()

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

    generate(out_dir / "model.json", out_dir)
    mq4_files = list(out_dir.glob("Generated_*.mq4"))
    assert mq4_files, "EA file not generated"
    text = mq4_files[0].read_text()
    assert "GraphDegree()" in text
    assert "GraphPagerank()" in text

