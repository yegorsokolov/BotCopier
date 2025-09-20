import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

import botcopier.features.technical as technical
from botcopier.features.engineering import FeatureConfig, configure_cache
from botcopier.models.schema import ModelParams


def _build_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("SYM", type="symbol")
    g.add_node("COMP", type="entity")
    g.add_node("SEC", type="sector")
    g.add_node("A1", type="article", sentiment=0.1)
    g.add_edge("SYM", "COMP")
    g.add_edge("COMP", "SEC")
    g.add_edge("A1", "COMP")
    return g


def test_graph_features_update(tmp_path: Path) -> None:
    df = pd.DataFrame({"symbol": ["SYM"], "price": [1.0]})
    g = _build_graph()
    config = configure_cache(FeatureConfig())
    feats: list[str] = []
    out, feats1, *_ = technical._extract_features(
        df.copy(), feats, entity_graph=g, config=config
    )
    assert out["graph_article_count"].iloc[0] == 1
    assert out["graph_sentiment"].iloc[0] == 0.1
    assert "graph_article_count" in feats1
    assert "graph_sentiment" in feats1

    # Add a new entity and article connected via the sector
    g.add_node("COMP2", type="entity")
    g.add_edge("COMP2", "SEC")
    g.add_node("A2", type="article", sentiment=0.6)
    g.add_edge("A2", "COMP2")

    out2, feats2, *_ = technical._extract_features(
        df.copy(), [], entity_graph=g, config=config
    )
    assert out2["graph_article_count"].iloc[0] == 2
    assert out2["graph_sentiment"].iloc[0] == pytest.approx(0.35)

    # Graph snapshot serialized into model.json
    params = ModelParams(feature_names=feats2, graph_snapshot=technical._GRAPH_SNAPSHOT)
    model_path = tmp_path / "model.json"
    model_path.write_text(params.model_dump_json())
    data = json.loads(model_path.read_text())
    assert "graph_snapshot" in data
    assert any(n.get("id") == "SYM" for n in data["graph_snapshot"]["nodes"])
