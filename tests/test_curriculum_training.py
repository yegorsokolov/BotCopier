import pytest

np = pytest.importorskip("numpy")

from botcopier.training.curriculum import _apply_curriculum


def test_curriculum_progression():
    amplitudes = np.concatenate([
        np.full(30, 0.1),
        np.full(30, 1.0),
        np.full(30, 5.0),
    ])
    signs = np.concatenate([
        np.repeat([1, -1], 15),
        np.ones(30),
        np.repeat([1, -1], 15),
    ])
    profits = amplitudes
    # Two-feature representation so that ``np.std`` captures volatility while the
    # sign of the second feature determines the label.
    X = np.column_stack([np.zeros_like(amplitudes), amplitudes * signs])
    y = (signs > 0).astype(float)
    sample_weight = amplitudes
    X_sel, y_sel, prof_sel, w_sel, R_sel, meta = _apply_curriculum(
        X,
        y,
        profits,
        sample_weight,
        model_type="logreg",
        gpu_kwargs={},
        grad_clip=1.0,
        threshold=1e-6,
        steps=3,
    )
    assert len(meta) == 3
    vals = [m["val_profit"] for m in meta]
    assert vals[0] < vals[1] < vals[2]
    assert len(prof_sel) == len(profits)
    assert R_sel is None
