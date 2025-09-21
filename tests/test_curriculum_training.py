import pytest

np = pytest.importorskip("numpy")

from botcopier.training import curriculum as curriculum_module
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


def test_crossmodal_sequence_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    profits = np.linspace(0.1, 0.6, 6, dtype=float)
    X = np.column_stack([profits, profits * 2.0])
    y = (profits > 0.3).astype(float)
    sample_weight = np.ones_like(profits)
    captured_kwargs: list[dict[str, object]] = []

    def fake_get_model(name: str):
        assert name == "crossmodal"

        def builder(X_train, y_train, **kwargs):
            captured_kwargs.append(dict(kwargs))

            def pred_fn(X_val):
                return np.zeros(len(X_val))

            return object(), pred_fn

        return builder

    monkeypatch.setattr(curriculum_module, "get_model", fake_get_model)
    gpu_kwargs = {"device": "cuda"}

    _apply_curriculum(
        X,
        y,
        profits,
        sample_weight,
        model_type="crossmodal",
        gpu_kwargs=gpu_kwargs,
        grad_clip=0.5,
        threshold=1e-3,
        steps=2,
    )

    assert captured_kwargs
    assert gpu_kwargs == {"device": "cuda"}
    for kwargs in captured_kwargs:
        assert kwargs["grad_clip"] == 0.5
        assert "sample_weight" not in kwargs
        assert kwargs["device"] == "cuda"
