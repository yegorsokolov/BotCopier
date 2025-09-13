import pytest

np = pytest.importorskip("numpy")

from botcopier.training.curriculum import _apply_curriculum

def test_curriculum_progression():
    magnitudes = np.concatenate([
        np.full(30, 0.001),
        np.full(30, 0.01),
        np.full(30, 0.1),
    ])
    signs = np.concatenate([
        np.repeat([1, -1], 15),
        np.repeat([1, -1], 15),
        np.repeat([1, -1], 15),
    ])
    profits = magnitudes
    X = (signs * 10).reshape(-1, 1)
    y = (signs > 0).astype(float)
    sample_weight = magnitudes
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
