from pathlib import Path
import numpy as np

from botcopier.data.loading import _load_logs
from botcopier.features.engineering import (
    FeatureConfig,
    _extract_features,
    configure_cache,
)


def test_pandas_dask_equivalence():
    data = Path("tests/fixtures/trades_small.csv")
    pdf, feats, _ = _load_logs(data, feature_config=configure_cache(FeatureConfig()))
    ddf, feats2, _ = _load_logs(
        data, dask=True, feature_config=configure_cache(FeatureConfig())
    )
    assert feats == feats2
    pdf_feat, feats, *_ = _extract_features(
        pdf.copy(), feats, config=configure_cache(FeatureConfig())
    )
    ddf_feat, feats, *_ = _extract_features(ddf, feats, config=configure_cache(FeatureConfig()))
    pd_vals = pdf_feat[feats].fillna(0).to_numpy()
    dd_vals = ddf_feat.compute()[feats].fillna(0).to_numpy()
    assert np.allclose(pd_vals, dd_vals)
