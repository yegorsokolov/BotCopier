import numpy as np
from botcopier.scripts.features import _kalman_filter_series, _kalman_update, KALMAN_DEFAULT_PARAMS


def python_kalman(values, params):
    state = {}
    lvl = []
    tr = []
    for v in values:
        l, t = _kalman_update(state, float(v), **params)
        lvl.append(l)
        tr.append(t)
    return np.array(lvl), np.array(tr)


def test_kalman_filter_series_matches_python():
    rng = np.random.default_rng(0)
    values = rng.standard_normal(128).astype(float)
    params = KALMAN_DEFAULT_PARAMS
    lvl_nb, tr_nb = _kalman_filter_series(values, params["process_var"], params["measurement_var"])
    lvl_py, tr_py = python_kalman(values, params)
    assert np.allclose(lvl_nb, lvl_py)
    assert np.allclose(tr_nb, tr_py)
