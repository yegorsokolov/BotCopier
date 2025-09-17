"""Model builders and registry for BotCopier."""

from __future__ import annotations

import base64
import json
import logging
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
from opentelemetry import trace
from pydantic import ValidationError
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from botcopier.exceptions import ModelError
from .schema import ModelParams

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

try:  # Optional dependency
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - optional
    logger.exception("PyTorch is unavailable")
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # Optional dependency
    import xgboost as xgb  # type: ignore

    _HAS_XGB = True
except ImportError:  # pragma: no cover - optional
    logger.exception("XGBoost is unavailable")
    xgb = None  # type: ignore
    _HAS_XGB = False

try:  # Optional dependency
    import catboost as cb  # type: ignore

    _HAS_CATBOOST = True
except ImportError:  # pragma: no cover - optional
    logger.exception("CatBoost is unavailable")
    cb = None  # type: ignore
    _HAS_CATBOOST = False

MODEL_REGISTRY: Dict[str, Callable] = {}

MODEL_VERSION = ModelParams.model_fields["version"].default
MIGRATIONS: Dict[int, Callable[[dict], dict]] = {}


def register_migration(version: int, fn: Callable[[dict], dict]) -> None:
    """Register ``fn`` to upgrade from ``version`` to ``version + 1``."""

    MIGRATIONS[version] = fn


def _migrate_data(data: dict) -> dict:
    version = data.get("version", 0)
    while version < MODEL_VERSION:
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            break
        data = migrate(data)
        version = data.get("version", version + 1)
    data["version"] = MODEL_VERSION
    return data


def load_params(path: Path) -> ModelParams:
    """Load ``ModelParams`` from ``path`` upgrading older versions."""
    with tracer.start_as_current_span("load_params"):
        try:
            if path.suffix == ".gz":
                import gzip

                with gzip.open(path, "rt") as fh:
                    raw = fh.read()
            else:
                raw = path.read_text()
            try:
                params = ModelParams.model_validate_json(raw)
            except ValidationError:
                data = json.loads(raw)
                data = _migrate_data(data)
                params = ModelParams(**data)
            else:
                if params.version != MODEL_VERSION:
                    data = _migrate_data(params.model_dump())
                    params = ModelParams(**data)
            path.write_text(params.model_dump_json())
            return params
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            logger.exception("Failed to load model parameters from %s", path)
            raise ModelError("Failed to load model parameters") from exc


def register_model(name: str, builder: Callable) -> None:
    """Register ``builder`` under ``name``."""

    MODEL_REGISTRY[name] = builder


def get_model(name: str) -> Callable:
    """Retrieve a registered model builder."""

    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Model '{name}' is not registered") from exc


class _FeatureClipper(BaseEstimator, TransformerMixin):
    """Clip features to provided ``low`` and ``high`` bounds."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "_FeatureClipper":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.low, self.high)


def _extract_logreg_metadata(pipeline: Pipeline) -> dict[str, object]:
    """Return metadata describing a fitted logistic regression pipeline."""

    clf: LogisticRegression = pipeline.named_steps["logreg"]
    scaler: RobustScaler = pipeline.named_steps["scale"]
    clipper: _FeatureClipper = pipeline.named_steps["clip"]
    return {
        "coefficients": clf.coef_.ravel().tolist(),
        "intercept": float(clf.intercept_[0]),
        "feature_mean": scaler.center_.tolist(),
        "feature_std": scaler.scale_.tolist(),
        "clip_low": clipper.low.tolist(),
        "clip_high": clipper.high.tolist(),
    }


class ConfidenceWeighted:
    """Simple diagonal Confidence-Weighted classifier.

    The implementation follows the AROW update rule maintaining a mean
    weight vector and a diagonal covariance matrix.  It exposes a minimal
    scikit-learn like interface with ``partial_fit`` and ``predict``
    methods allowing it to be used by the online trainer.
    """

    def __init__(self, r: float = 1.0) -> None:
        self.r = r
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.sigma: np.ndarray | None = None
        self.bias_sigma: float = 1.0 / r
        self.classes_: np.ndarray | None = None
        self.last_batch_confidence: float | None = None

    def _ensure_init(self, n_features: int, classes: np.ndarray | None) -> None:
        if self.w is None:
            self.w = np.zeros(n_features, dtype=float)
            self.sigma = np.ones(n_features, dtype=float) / self.r
            self.bias_sigma = 1.0 / self.r
            if classes is not None:
                self.classes_ = np.asarray(classes)

    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, classes: np.ndarray | None = None
    ) -> "ConfidenceWeighted":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._ensure_init(X.shape[1], classes)
        assert self.w is not None and self.sigma is not None
        conf: list[float] = []
        for xi, yi in zip(X, y):
            yi2 = 1 if yi == 1 else -1
            wx = np.dot(self.w, xi) + self.b
            m = yi2 * wx
            v = float(np.dot(xi * xi, self.sigma) + self.bias_sigma)
            beta = 1.0 / (v + self.r)
            alpha = max(0.0, 1.0 - m) * beta
            self.w += alpha * yi2 * self.sigma * xi
            self.b += alpha * yi2 * self.bias_sigma
            # keep variances constant to allow adaptation
            conf.append(m / (np.sqrt(v) + 1e-12))
        if conf:
            self.last_batch_confidence = float(np.mean(conf))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        margin = X @ self.w + self.b
        probs1 = 1.0 / (1.0 + np.exp(-margin))
        return np.column_stack([1 - probs1, probs1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def confidence_score(self, X: np.ndarray) -> np.ndarray:
        """Return normalised margin for ``X``."""
        assert self.w is not None and self.sigma is not None
        margin = X @ self.w + self.b
        var = X * X @ self.sigma + self.bias_sigma
        return margin / (np.sqrt(var) + 1e-12)


def _fit_logreg(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    sample_weight: np.ndarray | None = None,
    init_weights: Sequence[float] | None = None,
    max_iter: int = 1000,
) -> tuple[dict[str, list | float], Callable[[np.ndarray], np.ndarray]]:
    """Fit logistic regression within a preprocessing pipeline.

    Parameters
    ----------
    C:
        Inverse regularisation strength passed to :class:`~sklearn.linear_model.LogisticRegression`.
    """

    clip_low = np.quantile(X, 0.01, axis=0)
    clip_high = np.quantile(X, 0.99, axis=0)
    logreg = LogisticRegression(
        max_iter=max_iter, C=C, warm_start=init_weights is not None
    )
    pipeline = Pipeline(
        [
            ("clip", _FeatureClipper(clip_low, clip_high)),
            ("scale", RobustScaler()),
            ("logreg", logreg),
        ]
    )
    if init_weights is not None:
        logreg.classes_ = np.array([0, 1])
        logreg.coef_ = np.asarray(init_weights, dtype=float).reshape(1, -1)
        logreg.intercept_ = np.zeros(1)
    fit_kwargs = (
        {"logreg__sample_weight": sample_weight} if sample_weight is not None else {}
    )
    pipeline.fit(X, y, **fit_kwargs)

    def _predict(arr: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(arr)[:, 1]

    _predict.model = pipeline  # type: ignore[attr-defined]

    meta = _extract_logreg_metadata(pipeline)
    meta["training_rows"] = int(X.shape[0])
    cv_acc = pipeline.score(X, y)
    meta["cv_accuracy"] = float(cv_acc)
    return meta, _predict


def _fit_confidence_weighted(
    X: np.ndarray,
    y: np.ndarray,
    *,
    r: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> tuple[dict[str, object], Callable[[np.ndarray], np.ndarray]]:
    """Fit a :class:`ConfidenceWeighted` classifier on ``X`` and ``y``."""

    clf = ConfidenceWeighted(r=r)
    clf.partial_fit(X, y, classes=np.array([0, 1]))

    def _predict(arr: np.ndarray) -> np.ndarray:
        return clf.predict_proba(arr)[:, 1]

    _predict.model = clf  # type: ignore[attr-defined]
    meta = {
        "coefficients": clf.w.tolist() if clf.w is not None else [],
        "intercept": float(clf.b),
        "variance": clf.sigma.tolist() if clf.sigma is not None else [],
        "bias_variance": float(clf.bias_sigma),
        "model_type": "confidence_weighted",
    }
    return meta, _predict


register_model("logreg", _fit_logreg)
register_model("confidence_weighted", _fit_confidence_weighted)


def _fit_gradient_boosting_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    random_state: int | None = None,
    **params: float | int | str,
) -> tuple[dict[str, object], Callable[[np.ndarray], np.ndarray]]:
    """Fit a :class:`~sklearn.ensemble.GradientBoostingClassifier`."""

    default_params: dict[str, object] = {
        "learning_rate": 0.1,
        "n_estimators": 100,
        "max_depth": 3,
    }
    if random_state is not None:
        default_params.setdefault("random_state", random_state)
    default_params.update(params)
    model = GradientBoostingClassifier(**default_params)
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(X, y, **fit_kwargs)

    def _predict(arr: np.ndarray) -> np.ndarray:
        return model.predict_proba(arr)[:, 1]

    _predict.model = model  # type: ignore[attr-defined]

    serialised = base64.b64encode(pickle.dumps(model)).decode("utf-8")
    meta = {
        "gb_params": model.get_params(),
        "gb_model": serialised,
    }
    return meta, _predict


def _fit_voting_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    weights: Sequence[float] | None = None,
    include_xgboost: bool | None = None,
    C: float = 1.0,
    max_iter: int = 1000,
    gb_params: dict[str, object] | None = None,
    xgb_params: dict[str, object] | None = None,
    random_state: int | None = None,
) -> tuple[dict[str, object], Callable[[np.ndarray], np.ndarray]]:
    """Train a soft-voting ensemble combining linear and tree models."""

    clip_low = np.quantile(X, 0.01, axis=0)
    clip_high = np.quantile(X, 0.99, axis=0)
    logreg = LogisticRegression(max_iter=max_iter, C=C)
    pipeline = Pipeline(
        [
            ("clip", _FeatureClipper(clip_low, clip_high)),
            ("scale", RobustScaler()),
            ("logreg", logreg),
        ]
    )
    gb_defaults: dict[str, object] = {
        "learning_rate": 0.1,
        "n_estimators": 100,
        "max_depth": 3,
    }
    if random_state is not None:
        gb_defaults.setdefault("random_state", random_state)
    gb_defaults.update(gb_params or {})
    gbrt = GradientBoostingClassifier(**gb_defaults)
    estimators: list[tuple[str, object]] = [("logreg", pipeline), ("gbrt", gbrt)]
    estimator_meta: list[dict[str, object]] = []
    use_xgb = include_xgboost if include_xgboost is not None else _HAS_XGB
    xgb_model = None
    if use_xgb and _HAS_XGB:
        xgb_defaults: dict[str, object] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
        }
        xgb_defaults.update(xgb_params or {})
        xgb_model = xgb.XGBClassifier(**xgb_defaults)
        estimators.append(("xgb", xgb_model))

    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=list(weights) if weights is not None else None,
        flatten_transform=False,
    )
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    ensemble.fit(X, y, **fit_kwargs)

    def _predict(arr: np.ndarray) -> np.ndarray:
        return ensemble.predict_proba(arr)[:, 1]

    _predict.model = ensemble  # type: ignore[attr-defined]

    fitted_logreg: Pipeline = ensemble.named_estimators_["logreg"]  # type: ignore[index]
    logreg_meta = _extract_logreg_metadata(fitted_logreg)
    estimator_meta.append(
        {
            "name": "logreg",
            "type": "logistic",
            "coefficients": logreg_meta["coefficients"],
            "intercept": logreg_meta["intercept"],
            "clip_low": logreg_meta["clip_low"],
            "clip_high": logreg_meta["clip_high"],
            "center": logreg_meta["feature_mean"],
            "scale": logreg_meta["feature_std"],
        }
    )
    fitted_gbrt: GradientBoostingClassifier = ensemble.named_estimators_["gbrt"]  # type: ignore[index]
    gb_serialised = base64.b64encode(pickle.dumps(fitted_gbrt)).decode("utf-8")
    estimator_meta.append(
        {
            "name": "gradient_boosting",
            "type": "gradient_boosting",
            "model": gb_serialised,
            "params": fitted_gbrt.get_params(),
        }
    )
    if use_xgb and xgb_model is not None:
        fitted_xgb = ensemble.named_estimators_.get("xgb")
        if fitted_xgb is not None:
            booster_bytes = fitted_xgb.get_booster().save_raw()
            estimator_meta.append(
                {
                    "name": "xgboost",
                    "type": "xgboost",
                    "booster": base64.b64encode(booster_bytes).decode("utf-8"),
                    "params": fitted_xgb.get_params(),
                }
            )

    ensemble_info: dict[str, object] = {
        "type": "soft_voting",
        "weights": [float(w) for w in weights] if weights is not None else None,
        "estimators": estimator_meta,
    }

    meta: dict[str, object] = {
        "coefficients": logreg_meta["coefficients"],
        "intercept": logreg_meta["intercept"],
        "entry_coefficients": logreg_meta["coefficients"],
        "entry_intercept": logreg_meta["intercept"],
        "feature_mean": logreg_meta["feature_mean"],
        "feature_std": logreg_meta["feature_std"],
        "clip_low": logreg_meta["clip_low"],
        "clip_high": logreg_meta["clip_high"],
        "ensemble": ensemble_info,
    }
    return meta, _predict


register_model("gradient_boosting", _fit_gradient_boosting_classifier)
register_model("ensemble_voting", _fit_voting_ensemble)

if _HAS_TORCH:

    from torch.utils.data import DataLoader, TensorDataset

    from .deep import TabTransformer, TCNClassifier

    def _fit_moe(
        X: np.ndarray,
        y: np.ndarray,
        *,
        regime_features: np.ndarray,
        regime_feature_names: Sequence[str],
        n_experts: int | None = None,
        epochs: int = 50,
        lr: float = 1e-2,
        grad_clip: float = 1.0,
        sample_weight: np.ndarray | None = None,
        device: str = "cpu",
        dropout: float = 0.0,
        init_weights: object | None = None,
    ) -> tuple[dict[str, object], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Train a Mixture-of-Experts model on ``X`` and ``y``."""

        n_experts = n_experts or regime_features.shape[1]
        dev = torch.device(device)
        n_features = int(X.shape[1])
        regime_dim = int(regime_features.shape[1])
        model = MixtureOfExperts(
            n_features,
            regime_dim,
            n_experts,
            dropout=dropout,
        ).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sw: torch.Tensor | None = None
        if sample_weight is not None:
            sw = torch.tensor(sample_weight, dtype=torch.float32, device=dev).unsqueeze(
                1
            )
            loss_fn = torch.nn.BCELoss(weight=sw)
        else:
            loss_fn = torch.nn.BCELoss()

        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        R_t = torch.tensor(regime_features, dtype=torch.float32, device=dev)
        y_t = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(1)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            out, _ = model(X_t, R_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        model.to(dev)
        model.eval()

        def _predict(arr: np.ndarray, reg: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                arr_t = torch.tensor(arr, dtype=torch.float32, device=dev)
                reg_t = torch.tensor(reg, dtype=torch.float32, device=dev)
                out, _ = model(arr_t, reg_t)
                return out.squeeze(1).cpu().numpy()

        _predict.model = model  # type: ignore[attr-defined]

        feature_clip_low = np.quantile(X, 0.01, axis=0)
        feature_clip_high = np.quantile(X, 0.99, axis=0)
        feature_clipped = np.clip(X, feature_clip_low, feature_clip_high)
        feature_mean = feature_clipped.mean(axis=0)
        feature_std = feature_clipped.std(axis=0)

        state_dict = {k: v.numpy().tolist() for k, v in state.items()}
        meta = {
            "experts": [
                {
                    "weights": e.weight.detach().cpu().numpy().ravel().tolist(),
                    "bias": float(e.bias.detach().cpu().item()),
                }
                for e in model.experts
            ],
            "regime_gating": {
                "weights": model.gating.weight.detach().cpu().numpy().tolist(),
                "bias": model.gating.bias.detach().cpu().numpy().tolist(),
                "classes": list(range(n_experts)),
                "feature_names": list(regime_feature_names),
            },
            "state_dict": state_dict,
            "architecture": {
                "type": "MixtureOfExperts",
                "num_features": n_features,
                "regime_features": regime_dim,
                "n_experts": int(n_experts),
                "dropout": float(dropout),
            },
            "regime_features": list(regime_feature_names),
            "clip_low": feature_clip_low.tolist(),
            "clip_high": feature_clip_high.tolist(),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "model_type": "moe",
        }
        return meta, _predict

    register_model("moe", _fit_moe)

if _HAS_XGB:

    def _fit_xgboost_classifier(
        X: np.ndarray,
        y: np.ndarray,
        *,
        tree_method: str | None = None,
        predictor: str | None = None,
        sample_weight: np.ndarray | None = None,
        **params: float | int | str,
    ) -> tuple[dict[str, object], Callable[[np.ndarray], np.ndarray]]:
        """Fit an ``xgboost.XGBClassifier`` model.

        Parameters
        ----------
        tree_method:
            Optional ``tree_method`` passed to :class:`xgboost.XGBClassifier`.
            Passing ``"gpu_hist"`` enables GPU training when XGBoost is built
            with CUDA support.
        predictor:
            Optional ``predictor`` argument.  When ``tree_method`` starts with
            ``"gpu"`` and ``predictor`` is ``None`` this defaults to
            ``"gpu_predictor"``.
        """

        default_params: dict[str, object] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "verbosity": 0,
        }
        if tree_method is not None:
            default_params.setdefault("tree_method", tree_method)
            if tree_method.startswith("gpu") and predictor is None:
                predictor = "gpu_predictor"
        if predictor is not None:
            default_params.setdefault("predictor", predictor)
        default_params.update(params)
        model = xgb.XGBClassifier(**default_params)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X, y, **fit_kwargs)

        def _predict(arr: np.ndarray) -> np.ndarray:
            return model.predict_proba(arr)[:, 1]

        _predict.model = model  # type: ignore[attr-defined]
        booster_bytes = model.get_booster().save_raw()
        return {
            "booster": base64.b64encode(booster_bytes).decode("utf-8"),
            "xgb_params": model.get_params(),
        }, _predict

    register_model("xgboost", _fit_xgboost_classifier)
else:  # pragma: no cover - optional dependency

    def _fit_xgboost_classifier(*_, **__):  # type: ignore[dead-code]
        raise ImportError("xgboost is required for this model")


if _HAS_CATBOOST:

    def _fit_catboost_classifier(
        X: np.ndarray,
        y: np.ndarray,
        *,
        device: str | None = None,
        sample_weight: np.ndarray | None = None,
        **params: float | int | str,
    ) -> tuple[dict[str, object], Callable[[np.ndarray], np.ndarray]]:
        """Fit a ``catboost.CatBoostClassifier`` model.

        Parameters
        ----------
        device:
            Device for training.  Passing ``"gpu"`` enables GPU training when
            CatBoost is installed with CUDA support.
        """

        default_params: dict[str, object] = {"verbose": False}
        if device is not None:
            default_params.setdefault("device", device)
        default_params.update(params)
        model = cb.CatBoostClassifier(**default_params)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(X, y, **fit_kwargs)

        def _predict(arr: np.ndarray) -> np.ndarray:
            return model.predict_proba(arr)[:, 1]

        _predict.model = model  # type: ignore[attr-defined]
        return {"cat_params": model.get_params()}, _predict

    register_model("catboost", _fit_catboost_classifier)
else:  # pragma: no cover - optional dependency

    def _fit_catboost_classifier(*_, **__):  # type: ignore[dead-code]
        raise ImportError("catboost is required for this model")


if _HAS_TORCH:

    from torch.utils.data import DataLoader, TensorDataset

    from .deep import (
        CrossModalTransformer,
        MixtureOfExperts,
        SymbolContextAttention,
        TCNClassifier,
        TabTransformer,
    )

    def _sequence_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        flat = X.reshape(-1, X.shape[-1])
        clip_low = np.quantile(flat, 0.01, axis=0)
        clip_high = np.quantile(flat, 0.99, axis=0)
        clipped = np.clip(flat, clip_low, clip_high)
        mean = clipped.mean(axis=0)
        std = clipped.std(axis=0)
        return clip_low, clip_high, mean, std

    def _feature_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if X.ndim != 2:
            raise ValueError("expected 2D feature matrix")
        clip_low = np.quantile(X, 0.01, axis=0)
        clip_high = np.quantile(X, 0.99, axis=0)
        clipped = np.clip(X, clip_low, clip_high)
        mean = clipped.mean(axis=0)
        std = clipped.std(axis=0)
        return clip_low, clip_high, mean, std

    def _normalise_windows(
        X: np.ndarray,
        clip_low: np.ndarray,
        clip_high: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        std_safe = np.where(std == 0, 1.0, std)
        X_clip = np.clip(X, clip_low, clip_high)
        return (X_clip - mean) / std_safe

    def _train_sequence_model(
        model: torch.nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        grad_clip: float,
        patience: int,
        device: torch.device,
        mixed_precision: bool,
    ) -> dict[str, torch.Tensor]:
        if X.size == 0:
            raise ValueError("Empty training data")
        model.to(device)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        if sample_weight is not None:
            w_np = np.asarray(sample_weight, dtype=float)
        else:
            w_np = np.ones_like(y, dtype=float)
        w_t = torch.as_tensor(w_np, dtype=torch.float32, device=device)
        n_samples = X_t.shape[0]
        if n_samples == 1:
            train_X, val_X = X_t, X_t[:0]
            train_y, val_y = y_t, y_t[:0]
            train_w, val_w = w_t, w_t[:0]
        else:
            val_size = max(1, int(n_samples * 0.2))
            if val_size >= n_samples:
                val_size = n_samples - 1
            split = n_samples - val_size
            train_X, val_X = X_t[:split], X_t[split:]
            train_y, val_y = y_t[:split], y_t[split:]
            train_w, val_w = w_t[:split], w_t[split:]
        dataset = TensorDataset(train_X, train_y, train_w)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)) or 1,
            shuffle=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        amp_enabled = mixed_precision and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        def _autocast():
            if device.type == "cuda":
                return torch.cuda.amp.autocast(enabled=amp_enabled)
            return nullcontext()

        best_loss = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
        for _ in range(max(1, epochs)):
            model.train()
            for batch_X, batch_y, batch_w in loader:
                opt.zero_grad(set_to_none=True)
                with _autocast():
                    logits = model(batch_X)
                    batch_loss = loss_fn(logits, batch_y)
                    denom = batch_w.sum()
                    if denom.item() <= 0:
                        loss = batch_loss.mean()
                    else:
                        loss = (batch_loss * batch_w).sum() / denom
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            model.eval()
            with torch.no_grad():
                if val_X.numel() == 0:
                    val_loss = float(loss.item())
                else:
                    with _autocast():
                        logits = model(val_X)
                        val_raw = loss_fn(logits, val_y)
                    denom = val_w.sum()
                    if denom.item() <= 0:
                        val_loss = float(val_raw.mean().item())
                    else:
                        val_loss = float((val_raw * val_w).sum().item() / denom.item())
            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= max(1, patience):
                    break
        model.load_state_dict(best_state)
        model.eval()
        return best_state

    def _train_crossmodal_model(
        model: torch.nn.Module,
        price_seq: np.ndarray,
        news_seq: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        grad_clip: float,
        patience: int,
        device: torch.device,
        mixed_precision: bool,
    ) -> dict[str, torch.Tensor]:
        if price_seq.size == 0 or news_seq.size == 0:
            raise ValueError("Empty training data for cross-modal model")
        if price_seq.shape[0] != news_seq.shape[0]:
            raise ValueError("Price and news sequences must align")
        model.to(device)
        price_t = torch.as_tensor(price_seq, dtype=torch.float32, device=device)
        news_t = torch.as_tensor(news_seq, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
        if sample_weight is not None:
            w_np = np.asarray(sample_weight, dtype=float)
        else:
            w_np = np.ones_like(y, dtype=float)
        w_t = torch.as_tensor(w_np, dtype=torch.float32, device=device)
        n_samples = price_t.shape[0]
        if n_samples == 1:
            price_train, price_val = price_t, price_t[:0]
            news_train, news_val = news_t, news_t[:0]
            y_train, y_val = y_t, y_t[:0]
            w_train, w_val = w_t, w_t[:0]
        else:
            val_size = max(1, int(n_samples * 0.2))
            if val_size >= n_samples:
                val_size = n_samples - 1
            split = n_samples - val_size
            price_train, price_val = price_t[:split], price_t[split:]
            news_train, news_val = news_t[:split], news_t[split:]
            y_train, y_val = y_t[:split], y_t[split:]
            w_train, w_val = w_t[:split], w_t[split:]
        dataset = TensorDataset(price_train, news_train, y_train, w_train)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)) or 1,
            shuffle=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        amp_enabled = mixed_precision and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        def _autocast():
            if device.type == "cuda":
                return torch.cuda.amp.autocast(enabled=amp_enabled)
            return nullcontext()

        best_loss = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
        for _ in range(max(1, epochs)):
            model.train()
            for price_batch, news_batch, y_batch, w_batch in loader:
                opt.zero_grad(set_to_none=True)
                with _autocast():
                    logits = model(price_batch, news_batch)
                    batch_loss = loss_fn(logits, y_batch)
                    denom = w_batch.sum()
                    if denom.item() <= 0:
                        loss = batch_loss.mean()
                    else:
                        loss = (batch_loss * w_batch).sum() / denom
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            model.eval()
            with torch.no_grad():
                if price_val.numel() == 0:
                    val_loss = float(loss.item())
                else:
                    with _autocast():
                        logits = model(price_val, news_val)
                        val_raw = loss_fn(logits, y_val)
                    denom = w_val.sum()
                    if denom.item() <= 0:
                        val_loss = float(val_raw.mean().item())
                    else:
                        val_loss = float((val_raw * w_val).sum().item() / denom.item())
            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= max(1, patience):
                    break
        model.load_state_dict(best_state)
        model.eval()
        return best_state

    def fit_tab_transformer(
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        ff_dim: int = 128,
        patience: int = 5,
        grad_clip: float = 1.0,
        device: str = "cpu",
        mixed_precision: bool = True,
    ) -> tuple[dict[str, object], Callable[[np.ndarray], np.ndarray]]:
        seq = np.asarray(X, dtype=float)
        if seq.ndim == 2:
            seq = seq[:, np.newaxis, :]
        window = seq.shape[1]
        features = seq.shape[-1]
        clip_low, clip_high, mean, std = _sequence_stats(seq)
        seq_norm = _normalise_windows(seq, clip_low, clip_high, mean, std)
        dev = torch.device(device)
        model = TabTransformer(
            features,
            window,
            dim=dim,
            depth=depth,
            heads=heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        state = _train_sequence_model(
            model,
            seq_norm,
            np.asarray(y, dtype=float),
            sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            patience=patience,
            device=dev,
            mixed_precision=mixed_precision,
        )
        model.load_state_dict(state)
        model.to(dev)

        std_safe = np.where(std == 0, 1.0, std)

        def _predict(arr: np.ndarray) -> np.ndarray:
            data = np.asarray(arr, dtype=float)
            if data.ndim == 2:
                data = data[:, np.newaxis, :]
            norm = (np.clip(data, clip_low, clip_high) - mean) / std_safe
            with torch.no_grad():
                tens = torch.as_tensor(norm, dtype=torch.float32, device=dev)
                logits = model(tens)
                return torch.sigmoid(logits).cpu().numpy()

        _predict.model = model  # type: ignore[attr-defined]
        _predict.device = dev  # type: ignore[attr-defined]

        meta = {
            "state_dict": {k: v.numpy().tolist() for k, v in state.items()},
            "clip_low": clip_low.tolist(),
            "clip_high": clip_high.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "window": int(window),
            "config": {
                "dim": dim,
                "depth": depth,
                "heads": heads,
                "ff_dim": ff_dim,
                "dropout": dropout,
            },
            "architecture": {
                "type": "TabTransformer",
                "num_features": int(features),
                "window": int(window),
                "dim": int(dim),
                "depth": int(depth),
                "heads": int(heads),
                "ff_dim": int(ff_dim),
                "dropout": float(dropout),
            },
            "sequence_order": list(range(window)),
        }
        return meta, _predict

    def fit_temporal_cnn(
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        channels: Sequence[int] | None = None,
        kernel_size: int = 3,
        patience: int = 5,
        grad_clip: float = 1.0,
        device: str = "cpu",
        mixed_precision: bool = True,
    ) -> tuple[dict[str, object], Callable[[np.ndarray], np.ndarray]]:
        seq = np.asarray(X, dtype=float)
        if seq.ndim == 2:
            seq = seq[:, np.newaxis, :]
        window = seq.shape[1]
        features = seq.shape[-1]
        clip_low, clip_high, mean, std = _sequence_stats(seq)
        seq_norm = _normalise_windows(seq, clip_low, clip_high, mean, std)
        dev = torch.device(device)
        model = TCNClassifier(
            features,
            window,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        state = _train_sequence_model(
            model,
            seq_norm,
            np.asarray(y, dtype=float),
            sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            patience=patience,
            device=dev,
            mixed_precision=mixed_precision,
        )
        model.load_state_dict(state)
        model.to(dev)

        std_safe = np.where(std == 0, 1.0, std)

        def _predict(arr: np.ndarray) -> np.ndarray:
            data = np.asarray(arr, dtype=float)
            if data.ndim == 2:
                data = data[:, np.newaxis, :]
            norm = (np.clip(data, clip_low, clip_high) - mean) / std_safe
            with torch.no_grad():
                tens = torch.as_tensor(norm, dtype=torch.float32, device=dev)
                logits = model(tens)
                return torch.sigmoid(logits).cpu().numpy()

        _predict.model = model  # type: ignore[attr-defined]
        _predict.device = dev  # type: ignore[attr-defined]

        cfg_channels = list(channels or (64, 64))
        meta = {
            "state_dict": {k: v.numpy().tolist() for k, v in state.items()},
            "clip_low": clip_low.tolist(),
            "clip_high": clip_high.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "window": int(window),
            "config": {
                "channels": cfg_channels,
                "kernel_size": kernel_size,
                "dropout": dropout,
            },
            "architecture": {
                "type": "TemporalConvNet",
                "num_features": int(features),
                "window": int(window),
                "channels": [int(c) for c in cfg_channels],
                "kernel_size": int(kernel_size),
                "dropout": float(dropout),
            },
            "sequence_order": list(range(window)),
        }
        return meta, _predict

    def fit_crossmodal_transformer(
        data: tuple[np.ndarray, np.ndarray] | list[np.ndarray],
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        ff_dim: int = 128,
        patience: int = 5,
        grad_clip: float = 1.0,
        device: str = "cpu",
        mixed_precision: bool = True,
    ) -> tuple[dict[str, object], Callable[[tuple[np.ndarray, np.ndarray]], np.ndarray]]:
        if not isinstance(data, (tuple, list)) or len(data) != 2:
            raise ValueError("Cross-modal model expects price and news sequences")
        price_seq = np.asarray(data[0], dtype=float)
        news_seq = np.asarray(data[1], dtype=float)
        if price_seq.ndim == 2:
            price_seq = price_seq[:, np.newaxis, :]
        if news_seq.ndim == 2:
            news_seq = news_seq[:, np.newaxis, :]
        if price_seq.shape[0] != news_seq.shape[0]:
            raise ValueError("Price and news sequences must align")

        price_clip_low, price_clip_high, price_mean, price_std = _sequence_stats(price_seq)
        news_clip_low, news_clip_high, news_mean, news_std = _sequence_stats(news_seq)
        price_norm = _normalise_windows(price_seq, price_clip_low, price_clip_high, price_mean, price_std)
        news_norm = _normalise_windows(news_seq, news_clip_low, news_clip_high, news_mean, news_std)

        price_window = price_norm.shape[1]
        news_window = news_norm.shape[1]
        price_features = price_norm.shape[-1]
        news_features = news_norm.shape[-1]

        dev = torch.device(device)
        model = CrossModalTransformer(
            price_features,
            news_features,
            price_window,
            news_window,
            dim=dim,
            depth=depth,
            heads=heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        state = _train_crossmodal_model(
            model,
            price_norm,
            news_norm,
            np.asarray(y, dtype=float),
            sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            patience=patience,
            device=dev,
            mixed_precision=mixed_precision,
        )
        model.load_state_dict(state)
        model.to(dev)

        price_std_safe = np.where(price_std == 0, 1.0, price_std)
        news_std_safe = np.where(news_std == 0, 1.0, news_std)

        def _predict(inputs: tuple[np.ndarray, np.ndarray] | list[np.ndarray]) -> np.ndarray:
            if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
                raise ValueError("Cross-modal predictor expects price and news inputs")
            price_arr = np.asarray(inputs[0], dtype=float)
            news_arr = np.asarray(inputs[1], dtype=float)
            if price_arr.ndim == 2:
                price_arr = price_arr[:, np.newaxis, :]
            if news_arr.ndim == 2:
                news_arr = news_arr[:, np.newaxis, :]
            price_norm_local = (
                np.clip(price_arr, price_clip_low, price_clip_high) - price_mean
            ) / price_std_safe
            news_norm_local = (
                np.clip(news_arr, news_clip_low, news_clip_high) - news_mean
            ) / news_std_safe
            with torch.no_grad():
                price_t = torch.as_tensor(price_norm_local, dtype=torch.float32, device=dev)
                news_t = torch.as_tensor(news_norm_local, dtype=torch.float32, device=dev)
                logits = model(price_t, news_t)
                return torch.sigmoid(logits).cpu().numpy()

        _predict.model = model  # type: ignore[attr-defined]
        _predict.device = dev  # type: ignore[attr-defined]

        meta = {
            "state_dict": {k: v.numpy().tolist() for k, v in state.items()},
            "clip_low": price_clip_low.tolist(),
            "clip_high": price_clip_high.tolist(),
            "mean": price_mean.tolist(),
            "std": price_std.tolist(),
            "feature_mean": price_mean.tolist(),
            "feature_std": price_std.tolist(),
            "news_clip_low": news_clip_low.tolist(),
            "news_clip_high": news_clip_high.tolist(),
            "news_mean": news_mean.tolist(),
            "news_std": news_std.tolist(),
            "news_window": int(news_window),
            "window": int(price_window),
            "config": {
                "dim": dim,
                "depth": depth,
                "heads": heads,
                "ff_dim": ff_dim,
                "dropout": dropout,
            },
            "architecture": {
                "type": "CrossModalTransformer",
                "price_features": int(price_features),
                "price_window": int(price_window),
                "news_features": int(news_features),
                "news_window": int(news_window),
                "dim": int(dim),
                "depth": int(depth),
                "heads": int(heads),
                "ff_dim": int(ff_dim),
                "dropout": float(dropout),
            },
            "sequence_order": list(range(price_window)),
            "news_sequence_order": list(range(news_window)),
        }
        return meta, _predict

    def fit_multi_symbol_attention(
        X: np.ndarray,
        y: np.ndarray,
        *,
        symbol_ids: Sequence[int],
        symbol_names: Sequence[str],
        embeddings: np.ndarray,
        neighbor_index: Sequence[Sequence[int]],
        sample_weight: np.ndarray | None = None,
        epochs: int = 30,
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        hidden_dim: int = 64,
        heads: int = 4,
        patience: int = 5,
        grad_clip: float = 1.0,
        device: str = "cpu",
        mixed_precision: bool = True,
    ) -> tuple[dict[str, object], Callable[[tuple[np.ndarray, np.ndarray]], np.ndarray]]:
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float)
        symbols = np.asarray(symbol_ids, dtype=int)
        emb_array = np.asarray(embeddings, dtype=float)
        if features.ndim != 2:
            raise ValueError("multi-symbol model expects a 2D feature matrix")
        if targets.ndim != 1:
            raise ValueError("targets must be 1-dimensional")
        if symbols.shape[0] != features.shape[0] or targets.shape[0] != features.shape[0]:
            raise ValueError("features, targets and symbol_ids must align")
        if emb_array.ndim != 2 or emb_array.shape[0] != len(symbol_names):
            raise ValueError("embeddings must match provided symbol_names")
        if len(neighbor_index) != len(symbol_names):
            raise ValueError("neighbor_index must align with symbol_names")

        clip_low, clip_high, mean, std = _feature_stats(features)
        std_safe = np.where(std == 0, 1.0, std)
        norm_features = (np.clip(features, clip_low, clip_high) - mean) / std_safe

        sanitized_neighbors: list[list[int]] = []
        for idx, neigh in enumerate(neighbor_index):
            unique: list[int] = []
            for n in neigh:
                if 0 <= int(n) < len(symbol_names) and int(n) not in unique:
                    unique.append(int(n))
            if idx not in unique:
                unique.insert(0, idx)
            sanitized_neighbors.append(unique)

        dev = torch.device(device)
        model = SymbolContextAttention(
            norm_features.shape[1],
            torch.as_tensor(emb_array, dtype=torch.float32),
            sanitized_neighbors,
            heads=heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        model.to(dev)

        features_t = torch.as_tensor(norm_features, dtype=torch.float32)
        symbols_t = torch.as_tensor(symbols, dtype=torch.long)
        targets_t = torch.as_tensor(targets, dtype=torch.float32)
        if sample_weight is not None:
            weight_np = np.asarray(sample_weight, dtype=float)
        else:
            weight_np = np.ones_like(targets, dtype=float)
        weights_t = torch.as_tensor(weight_np, dtype=torch.float32)

        n_samples = features_t.size(0)
        if n_samples == 0:
            raise ValueError("Empty training data")
        if n_samples == 1:
            train_slice = slice(0, 1)
            val_slice = slice(1, 1)
        else:
            val_size = max(1, int(n_samples * 0.2))
            if val_size >= n_samples:
                val_size = n_samples - 1
            split = n_samples - val_size
            train_slice = slice(0, split)
            val_slice = slice(split, n_samples)

        dataset = TensorDataset(
            features_t[train_slice],
            symbols_t[train_slice],
            targets_t[train_slice],
            weights_t[train_slice],
        )
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)) or 1,
            shuffle=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        amp_enabled = mixed_precision and dev.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        def _autocast():
            if dev.type == "cuda":
                return torch.cuda.amp.autocast(enabled=amp_enabled)
            return nullcontext()

        best_loss = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
        for _ in range(max(1, epochs)):
            model.train()
            for feat_b, sym_b, tgt_b, w_b in loader:
                feat_b = feat_b.to(dev)
                sym_b = sym_b.to(dev)
                tgt_b = tgt_b.to(dev)
                w_b = w_b.to(dev)
                opt.zero_grad(set_to_none=True)
                with _autocast():
                    logits = model(feat_b, sym_b)
                    loss_raw = loss_fn(logits, tgt_b)
                    denom = w_b.sum()
                    if denom.item() <= 0:
                        loss = loss_raw.mean()
                    else:
                        loss = (loss_raw * w_b).sum() / denom
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()

            model.eval()
            with torch.no_grad():
                if val_slice.start == val_slice.stop:
                    val_loss = float(loss.item())
                else:
                    feat_val = features_t[val_slice].to(dev)
                    sym_val = symbols_t[val_slice].to(dev)
                    tgt_val = targets_t[val_slice].to(dev)
                    w_val = weights_t[val_slice].to(dev)
                    with _autocast():
                        logits = model(feat_val, sym_val)
                        val_raw = loss_fn(logits, tgt_val)
                    denom = w_val.sum()
                    if denom.item() <= 0:
                        val_loss = float(val_raw.mean().item())
                    else:
                        val_loss = float((val_raw * w_val).sum().item() / denom.item())
            if val_loss + 1e-6 < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= max(1, patience):
                    break

        model.load_state_dict(best_state)
        model.to(dev)
        model.eval()

        def _predict(inputs: tuple[np.ndarray, np.ndarray] | list[np.ndarray]) -> np.ndarray:
            if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
                raise ValueError("multi-symbol predictor expects (features, symbol_ids)")
            feats = np.asarray(inputs[0], dtype=float)
            sym_local = np.asarray(inputs[1], dtype=int)
            if feats.ndim != 2:
                raise ValueError("features must be 2D")
            if feats.shape[0] != sym_local.shape[0]:
                raise ValueError("features and symbol_ids must align")
            norm = (np.clip(feats, clip_low, clip_high) - mean) / std_safe
            with torch.no_grad():
                feat_t = torch.as_tensor(norm, dtype=torch.float32, device=dev)
                sym_t = torch.as_tensor(sym_local, dtype=torch.long, device=dev)
                logits = model(feat_t, sym_t)
                return torch.sigmoid(logits).cpu().numpy()

        _predict.model = model  # type: ignore[attr-defined]
        _predict.device = dev  # type: ignore[attr-defined]

        max_neigh = max(len(n) for n in sanitized_neighbors)
        attn_sum = np.zeros((len(symbol_names), max_neigh), dtype=float)
        attn_count = np.zeros(len(symbol_names), dtype=float)
        if n_samples:
            full_loader = DataLoader(
                TensorDataset(features_t, symbols_t),
                batch_size=min(batch_size, n_samples) or 1,
                shuffle=False,
            )
            with torch.no_grad():
                for feat_b, sym_b in full_loader:
                    feat_b = feat_b.to(dev)
                    sym_b = sym_b.to(dev)
                    _, weights = model(feat_b, sym_b, return_attention=True)
                    attn_np = weights.squeeze(1).cpu().numpy()
                    for i, sym_idx in enumerate(sym_b.cpu().tolist()):
                        attn_sum[sym_idx, : attn_np.shape[1]] += attn_np[i]
                        attn_count[sym_idx] += 1

        attn_avg = np.zeros_like(attn_sum)
        for idx in range(len(symbol_names)):
            if attn_count[idx] > 0:
                attn_avg[idx] = attn_sum[idx] / attn_count[idx]
        neighbor_order = {
            symbol_names[i]: [symbol_names[j] for j in sanitized_neighbors[i]]
            for i in range(len(symbol_names))
        }
        attention_weights = {
            symbol_names[i]: attn_avg[i, : len(sanitized_neighbors[i])].tolist()
            for i in range(len(symbol_names))
        }

        meta = {
            "state_dict": {k: v.numpy().tolist() for k, v in best_state.items()},
            "clip_low": clip_low.tolist(),
            "clip_high": clip_high.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "attention_weights": attention_weights,
            "neighbor_order": neighbor_order,
            "symbols": list(symbol_names),
            "symbol_embeddings": {
                symbol_names[i]: emb_array[i].tolist() for i in range(len(symbol_names))
            },
            "neighbor_index": [list(map(int, n)) for n in sanitized_neighbors],
            "architecture": {
                "type": "SymbolContextAttention",
                "feature_dim": int(norm_features.shape[1]),
                "embedding_dim": int(emb_array.shape[1]) if emb_array.ndim == 2 else 0,
                "hidden_dim": int(hidden_dim),
                "heads": int(heads),
                "dropout": float(dropout),
            },
        }
        return meta, _predict

    register_model("tabtransformer", fit_tab_transformer)
    register_model("tcn", fit_temporal_cnn)
    register_model("transformer", fit_tab_transformer)
    register_model("crossmodal", fit_crossmodal_transformer)
    register_model("multi_symbol", fit_multi_symbol_attention)
else:  # pragma: no cover - torch optional

    class TabTransformer:  # type: ignore[misc]
        def __init__(self, *_, **__):  # pragma: no cover - trivial
            raise ImportError("PyTorch is required for TabTransformer")

    class TCNClassifier:  # type: ignore[misc]
        def __init__(self, *_, **__):  # pragma: no cover - trivial
            raise ImportError("PyTorch is required for TemporalConvNet")

    def fit_tab_transformer(*_, **__):  # pragma: no cover - trivial
        raise ImportError("PyTorch is required for TabTransformer")

    def fit_temporal_cnn(*_, **__):  # pragma: no cover - trivial
        raise ImportError("PyTorch is required for TemporalConvNet")

    def fit_crossmodal_transformer(*_, **__):  # pragma: no cover - trivial
        raise ImportError("PyTorch is required for CrossModalTransformer")

    def fit_multi_symbol_attention(*_, **__):  # pragma: no cover - trivial
        raise ImportError("PyTorch is required for SymbolContextAttention")


__all__ = [
    "TabTransformer",
    "TCNClassifier",
    "MixtureOfExperts",
    "SymbolContextAttention",
    "fit_tab_transformer",
    "fit_temporal_cnn",
    "fit_crossmodal_transformer",
    "fit_multi_symbol_attention",
    "register_model",
    "get_model",
    "MODEL_REGISTRY",
    "register_migration",
    "load_params",
]
