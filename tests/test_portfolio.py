import numpy as np
import pandas as pd

from botcopier.scripts.portfolio import hierarchical_risk_parity


def test_hrp_weights_sum_to_one():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(100, 4))
    df = pd.DataFrame(data, columns=["A", "B", "C", "D"])
    weights, _ = hierarchical_risk_parity(df)
    assert np.isclose(weights.sum(), 1.0)


def test_hrp_reduces_volatility():
    rng = np.random.default_rng(1)
    cov = np.array(
        [[0.1, 0.08, 0.02], [0.08, 0.1, 0.02], [0.02, 0.02, 0.1]]
    )
    mean = np.zeros(3)
    data = rng.multivariate_normal(mean, cov, size=1000)
    df = pd.DataFrame(data, columns=["X", "Y", "Z"])
    hrp_w, _ = hierarchical_risk_parity(df)
    eq_w = np.repeat(1 / 3, 3)
    hrp_vol = np.std(df.values @ hrp_w.values)
    eq_vol = np.std(df.values @ eq_w)
    assert hrp_vol < eq_vol
