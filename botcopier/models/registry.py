"""Model builders and registry for BotCopier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np
from pydantic import ValidationError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .schema import ModelParams

try:  # Optional dependency
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    _HAS_TORCH = False

try:  # Optional dependency
    import xgboost as xgb  # type: ignore

    _HAS_XGB = True
except Exception:  # pragma: no cover - optional
    xgb = None  # type: ignore
    _HAS_XGB = False

try:  # Optional dependency
    import catboost as cb  # type: ignore

    _HAS_CATBOOST = True
except Exception:  # pragma: no cover - optional
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
    clf: LogisticRegression = pipeline.named_steps["logreg"]
    scaler: RobustScaler = pipeline.named_steps["scale"]
    clipper: _FeatureClipper = pipeline.named_steps["clip"]

    def _predict(arr: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(arr)[:, 1]

    _predict.model = pipeline  # type: ignore[attr-defined]

    meta = {
        "coefficients": clf.coef_.ravel().tolist(),
        "intercept": float(clf.intercept_[0]),
        "training_rows": int(X.shape[0]),
        "feature_mean": scaler.center_.tolist(),
        "feature_std": scaler.scale_.tolist(),
        "clip_low": clipper.low.tolist(),
        "clip_high": clipper.high.tolist(),
    }
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

if _HAS_TORCH:

    class MixtureOfExperts(torch.nn.Module):
        """Simple mixture of experts with a softmax gating network."""

        def __init__(
            self,
            n_features: int,
            n_regime_features: int,
            n_experts: int,
        ) -> None:
            super().__init__()
            self.experts = torch.nn.ModuleList(
                [torch.nn.Linear(n_features, 1) for _ in range(n_experts)]
            )
            self.gating = torch.nn.Linear(n_regime_features, n_experts)

        def forward(
            self, x: torch.Tensor, r: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            gate_logits = self.gating(r)
            gate = torch.softmax(gate_logits, dim=1)
            expert_logits = torch.cat([e(x) for e in self.experts], dim=1)
            expert_prob = torch.sigmoid(expert_logits)
            out = (gate * expert_prob).sum(dim=1, keepdim=True)
            return out, gate

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
    ) -> tuple[dict[str, object], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Train a Mixture-of-Experts model on ``X`` and ``y``."""

        n_experts = n_experts or regime_features.shape[1]
        dev = torch.device(device)
        model = MixtureOfExperts(X.shape[1], regime_features.shape[1], n_experts).to(
            dev
        )
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

        def _predict(arr: np.ndarray, reg: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                arr_t = torch.tensor(arr, dtype=torch.float32, device=dev)
                reg_t = torch.tensor(reg, dtype=torch.float32, device=dev)
                out, _ = model(arr_t, reg_t)
                return out.squeeze(1).cpu().numpy()

        _predict.model = model  # type: ignore[attr-defined]

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
        import base64

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

    class _TTBlock(torch.nn.Module):
        """Attention block used by :class:`TabTransformer`."""

        def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float) -> None:
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(dim, ff_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(ff_dim, dim),
            )
            self.norm1 = torch.nn.LayerNorm(dim)
            self.norm2 = torch.nn.LayerNorm(dim)
            self.drop1 = torch.nn.Dropout(dropout)
            self.drop2 = torch.nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            attn, _ = self.attn(x, x, x)
            x = self.norm1(x + self.drop1(attn))
            ff = self.ff(x)
            return self.norm2(x + self.drop2(ff))

    class TabTransformer(torch.nn.Module):
        """Simple tabular transformer with multi-head attention layers."""

        def __init__(
            self,
            num_features: int,
            dim: int = 32,
            heads: int = 4,
            depth: int = 2,
            ff_dim: int = 64,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.embed = torch.nn.Linear(1, dim)
            self.layers = torch.nn.ModuleList(
                [_TTBlock(dim, heads, ff_dim, dropout) for _ in range(depth)]
            )
            self.norm = torch.nn.LayerNorm(dim)
            self.head = torch.nn.Linear(num_features * dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            x = self.embed(x.unsqueeze(-1))
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            x = x.reshape(x.size(0), -1)
            return self.head(x)

    def fit_tab_transformer(
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        dropout: float = 0.0,
        device: str = "cpu",
        grad_clip: float = 1.0,
    ) -> tuple[dict[str, list], Callable[[np.ndarray], np.ndarray]]:
        """Train a :class:`TabTransformer` on ``X`` and ``y``."""

        dev = torch.device(device)
        model = TabTransformer(X.shape[1], dropout=dropout).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        y_t = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(-1)
        model.train()
        for _ in range(epochs):  # pragma: no cover - quick epochs
            opt.zero_grad()
            out = model(X_t)
            loss = loss_fn(out, y_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        def _predict(arr: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                arr_t = torch.tensor(arr, dtype=torch.float32, device=dev)
                return torch.sigmoid(model(arr_t)).cpu().numpy().squeeze(-1)

        _predict.model = model  # type: ignore[attr-defined]
        _predict.device = dev  # type: ignore[attr-defined]

        state = model.state_dict()
        dim = model.embed.out_features
        qkv = state["layers.0.attn.in_proj_weight"].cpu()
        weights = {
            "q_weight": qkv[:dim].tolist(),
            "k_weight": qkv[dim : 2 * dim].tolist(),
            "v_weight": qkv[2 * dim :].tolist(),
            "out_weight": state["layers.0.attn.out_proj.weight"].cpu().tolist(),
            "pos_embed_weight": state["embed.weight"].cpu().tolist(),
        }
        return {"weights": weights, "dropout": dropout}, _predict

    register_model("transformer", fit_tab_transformer)
else:  # pragma: no cover - torch optional

    class TabTransformer:  # type: ignore[misc]
        def __init__(self, *_, **__):  # pragma: no cover - trivial
            raise ImportError("PyTorch is required for TabTransformer")

    def fit_tab_transformer(*_, **__):  # pragma: no cover - trivial
        raise ImportError("PyTorch is required for TabTransformer")


__all__ = [
    "TabTransformer",
    "fit_tab_transformer",
    "register_model",
    "get_model",
    "MODEL_REGISTRY",
    "register_migration",
    "load_params",
]
