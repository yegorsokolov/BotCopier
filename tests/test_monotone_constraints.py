import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from botcopier.models.registry import get_model


def _toy_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 2))
    logits = 1.5 * X[:, 0] - 1.8 * X[:, 1]
    y = (logits + 0.2 * rng.normal(size=X.shape[0]) > 0).astype(int)
    return X, y


def test_gradient_boosting_monotone_constraints_modify_predictions() -> None:
    X, y = _toy_dataset()
    builder = get_model("gradient_boosting")
    _, unconstrained = builder(X, y, random_state=0)
    _, constrained = builder(X, y, random_state=0, monotone_constraints=[1, -1])

    unconstrained_probs = unconstrained(X)
    constrained_probs = constrained(X)
    diff = np.max(np.abs(unconstrained_probs - constrained_probs))
    assert diff > 1e-6

    monotone_param = constrained.model.get_params().get("monotonic_cst")  # type: ignore[attr-defined]
    assert monotone_param is not None
    assert tuple(monotone_param) == (1, -1)

    with pytest.raises(ValueError, match="feature count"):
        builder(X, y, monotone_constraints=[1])


def test_xgboost_monotone_constraints_modify_predictions() -> None:
    pytest.importorskip("xgboost")
    X, y = _toy_dataset()
    builder = get_model("xgboost")
    _, unconstrained = builder(X, y, random_state=0, n_estimators=40)
    _, constrained = builder(
        X,
        y,
        random_state=0,
        n_estimators=40,
        monotone_constraints=[1, -1],
    )

    unconstrained_probs = unconstrained(X)
    constrained_probs = constrained(X)
    diff = np.max(np.abs(unconstrained_probs - constrained_probs))
    assert diff > 1e-6

    monotone_param = constrained.model.get_params().get("monotone_constraints")  # type: ignore[attr-defined]
    assert isinstance(monotone_param, str)
    assert monotone_param.replace(" ", "") == "(1,-1)"

    with pytest.raises(ValueError, match="feature count"):
        builder(X, y, monotone_constraints=[1])


def test_catboost_monotone_constraints_modify_predictions() -> None:
    pytest.importorskip("catboost")
    X, y = _toy_dataset()
    builder = get_model("catboost")
    _, unconstrained = builder(X, y, iterations=50, random_seed=0)
    _, constrained = builder(
        X,
        y,
        iterations=50,
        random_seed=0,
        monotone_constraints=[1, -1],
    )

    unconstrained_probs = unconstrained(X)
    constrained_probs = constrained(X)
    diff = np.max(np.abs(unconstrained_probs - constrained_probs))
    assert diff > 1e-6

    params = constrained.model.get_params()  # type: ignore[attr-defined]
    assert params.get("monotone_constraints") == [1, -1]

    with pytest.raises(ValueError, match="feature count"):
        builder(X, y, iterations=20, random_seed=0, monotone_constraints=[1])
