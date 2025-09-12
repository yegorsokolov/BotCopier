import pandas as pd
from botcopier.features import _extract_features
from botcopier.features.engineering import configure_cache, FeatureConfig


def test_orderbook_features_shapes_and_ranges():
    df = pd.DataFrame(
        {
            "bid": [100.0, 101.0, 102.0],
            "ask": [101.0, 102.0, 103.0],
            "bid_depth": [[5, 4], [7, 3], [8, 2]],
            "ask_depth": [[4, 5], [3, 6], [2, 7]],
        }
    )
    configure_cache(FeatureConfig(enabled_features={"orderbook"}))
    feats, cols, _, _ = _extract_features(df.copy(), [])
    configure_cache(FeatureConfig())  # reset for other tests

    assert {"depth_microprice", "depth_vol_imbalance", "depth_order_flow_imbalance"} <= set(
        cols
    )
    assert len(feats) == len(df)
    # microprice bounded by bid and ask
    assert ((feats["depth_microprice"] >= feats["bid"]) & (feats["depth_microprice"] <= feats["ask"])).all()
    # volume imbalance within [-1, 1]
    assert ((feats["depth_vol_imbalance"] <= 1) & (feats["depth_vol_imbalance"] >= -1)).all()
    # first order flow imbalance is zero
    assert abs(feats["depth_order_flow_imbalance"].iloc[0]) < 1e-9
