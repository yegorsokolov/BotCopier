import numpy as np
import pandas as pd

from scripts.meta_strategy import RollingMetrics, select_model, train_meta_model


def test_partial_fit_updates_coefficients_and_selection() -> None:
    # Initial dataset favours model 0 for positive features
    df1 = pd.DataFrame({"f1": [2, 3, -2, -3], "best_model": [0, 0, 1, 1]})
    params1 = train_meta_model(df1, ["f1"], use_partial_fit=True)
    coeff1 = np.array(params1["gating_coefficients"])
    # For feature value 4 the initial model should select model 0
    assert select_model(params1, {"f1": 4}) == 0

    # New data indicates model 1 should be chosen for large positive features
    df2 = pd.DataFrame({"f1": [4, 5, 6], "best_model": [1, 1, 1]})
    params2 = train_meta_model(df2, ["f1"], params=params1, use_partial_fit=True)
    coeff2 = np.array(params2["gating_coefficients"])

    # Coefficients should change after incremental update
    assert not np.allclose(coeff1, coeff2)
    # Updated model should now select model 1 for feature value 4
    assert select_model(params2, {"f1": 4}) == 1


def test_weighted_ensemble_adapts_to_performance() -> None:
    rm = RollingMetrics(n_models=2, alpha=0.5)
    preds = [0.2, 0.8]

    # Initially model 0 performs better
    rm.update(0, profit=1.0, correct=True)
    rm.update(1, profit=-1.0, correct=False)
    weights1 = rm.weights()
    out1 = rm.combine(preds)
    assert weights1[0] > weights1[1]

    # Model 1 starts to outperform
    rm.update(0, profit=-1.0, correct=False)
    rm.update(1, profit=2.0, correct=True)
    weights2 = rm.weights()
    out2 = rm.combine(preds)
    assert weights2[1] > weights2[0]
    # Ensemble output shifts toward model 1 prediction
    assert out2 > out1
