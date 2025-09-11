import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from botcopier.scripts.feature_inspect import (
    save_partial_dependence,
    save_permutation_importance,
)


def test_feature_importance_and_pdp_shapes(tmp_path):
    X, y = make_regression(n_samples=40, n_features=3, random_state=0)
    model = RandomForestRegressor(random_state=0).fit(X, y)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    out_dir = tmp_path / "reports/feature_analysis"

    ranks = save_permutation_importance(model, X, y, feature_names, out_dir)
    saved_ranks = np.load(out_dir / "importance_ranks.npy", allow_pickle=True)
    assert ranks.shape == (X.shape[1],)
    assert saved_ranks.shape == (X.shape[1],)

    pd_arrays = save_partial_dependence(model, X, [0, 1], feature_names, out_dir)
    saved_pdp = np.load(out_dir / "partial_dependence.npy")
    assert pd_arrays.shape[0] == 2
    assert saved_pdp.shape == pd_arrays.shape
    assert pd_arrays.shape[1] > 1
