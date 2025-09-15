import json
from pathlib import Path

import pandas as pd
import pytest

import botcopier.features.technical as technical
from sklearn.linear_model import LogisticRegression


@pytest.mark.skipif(not getattr(technical, "_HAS_TG", False), reason="torch_geometric not installed")
def test_gnn_embeddings_deterministic(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "price": [1.0, 1.1],
        "volume": [100, 110],
        "symbol": ["EURUSD", "USDCHF"],
    })
    sg_path = Path(__file__).resolve().parent.parent / "symbol_graph.json"
    out, feats1, emb1, gstate1 = technical._extract_features(df.copy(), [], symbol_graph=sg_path)
    assert any(c.startswith("graph_emb") for c in feats1)
    assert emb1
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps({"gnn_state": gstate1}))
    technical._GNN_STATE = None
    out2, feats2, emb2, gstate2 = technical._extract_features(
        df.copy(), [], symbol_graph=sg_path, gnn_state=gstate1
    )
    assert emb1 == emb2
    assert feats1 == feats2


@pytest.mark.skipif(not getattr(technical, "_HAS_TG", False), reason="torch_geometric not installed")
def test_gnn_embeddings_improve_accuracy() -> None:
    sg_path = Path(__file__).resolve().parent.parent / "symbol_graph.json"
    rows = []
    for i in range(20):
        t = f"2020-01-01T00:{i:02d}:00"
        rows.append(("EURUSD", 1.0, t, 1))
        rows.append(("USDCHF", 1.0, t, 0))
    df = pd.DataFrame(rows, columns=["symbol", "price", "event_time", "label"])
    df_no, feats_no, _, _ = technical._extract_features(df.copy(), [])
    X_no = df_no[feats_no].to_numpy(dtype=float)
    y = df_no["label"].to_numpy(dtype=float)
    acc_no = LogisticRegression().fit(X_no, y).score(X_no, y)
    df_g, feats_g, _, _ = technical._extract_features(df.copy(), [], symbol_graph=sg_path)
    X_g = df_g[feats_g].to_numpy(dtype=float)
    y_g = df_g["label"].to_numpy(dtype=float)
    acc_g = LogisticRegression().fit(X_g, y_g).score(X_g, y_g)
    assert acc_g > acc_no
    assert any(c.startswith("graph_emb") for c in feats_g)
    assert technical._GNN_STATE is not None
