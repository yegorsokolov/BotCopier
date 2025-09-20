import logging
import sys
import types

import pandas as pd

stub = types.SimpleNamespace(
    _augment_dataframe=lambda df, ratio, **_: df,
    _augment_dtw_dataframe=lambda df, ratio, **_: df,
)
sys.modules.setdefault("botcopier.features.augmentation", stub)

from botcopier.data.loading import _drop_duplicates_and_outliers, _load_logs
from botcopier.features.engineering import FeatureConfig, configure_cache


def _build_df() -> pd.DataFrame:
    times = pd.date_range("2020-01-01", periods=100, freq="T")
    base = pd.DataFrame({"event_time": times, "price": 1.0})
    dup = base.iloc[[0]]
    outlier_time = times[-1] + pd.Timedelta(minutes=1)
    outlier = pd.DataFrame({"event_time": [outlier_time], "price": [1000.0]})
    return pd.concat([dup, base, outlier], ignore_index=True)


def test_drop_duplicates_and_outliers():
    df = _build_df()
    cleaned, dup_cnt, out_cnt = _drop_duplicates_and_outliers(df)
    assert dup_cnt == 1
    assert out_cnt == 1
    assert len(cleaned) == 100


def test_load_logs_reports_sanitization(tmp_path, caplog):
    df = _build_df()
    file_path = tmp_path / "trades_raw.csv"
    df.to_csv(file_path, index=False)
    with caplog.at_level(logging.INFO):
        loaded, _, _ = _load_logs(
            file_path, feature_config=configure_cache(FeatureConfig())
        )
    assert len(loaded) == 100
    assert any(
        "Removed 1 duplicate rows and 1 outlier rows" in m for m in caplog.messages
    )
