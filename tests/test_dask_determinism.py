from pathlib import Path
import numpy as np

from botcopier.data.loading import _load_logs
from botcopier.features.technical import _extract_features


def test_pandas_dask_equivalence():
    data = Path("tests/fixtures/trades_small.csv")
    pdf, feats, _ = _load_logs(data)
    ddf, feats2, _ = _load_logs(data, dask=True)
    assert feats == feats2
    pdf_feat, feats, *_ = _extract_features(pdf.copy(), feats)
    ddf_feat, feats, *_ = _extract_features(ddf, feats)
    pd_vals = pdf_feat[feats].fillna(0).to_numpy()
    dd_vals = ddf_feat.compute()[feats].fillna(0).to_numpy()
    assert np.allclose(pd_vals, dd_vals)
