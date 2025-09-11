"""Model builders and registry for BotCopier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

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


def _fit_logreg(
    X: np.ndarray, y: np.ndarray, *, C: float = 1.0
) -> tuple[dict[str, list | float], Callable[[np.ndarray], np.ndarray]]:
    """Fit logistic regression within a preprocessing pipeline.

    Parameters
    ----------
    C:
        Inverse regularisation strength passed to :class:`~sklearn.linear_model.LogisticRegression`.
    """

    clip_low = np.quantile(X, 0.01, axis=0)
    clip_high = np.quantile(X, 0.99, axis=0)
    pipeline = Pipeline(
        [
            ("clip", _FeatureClipper(clip_low, clip_high)),
            ("scale", RobustScaler()),
            ("logreg", LogisticRegression(max_iter=1000, C=C)),
        ]
    )
    pipeline.fit(X, y)
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


register_model("logreg", _fit_logreg)

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
            opt.step()

        def _predict(arr: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                arr_t = torch.tensor(arr, dtype=torch.float32, device=dev)
                return torch.sigmoid(model(arr_t)).cpu().numpy().squeeze(-1)
        _predict.model = model  # type: ignore[attr-defined]
        _predict.device = dev  # type: ignore[attr-defined]

        state = model.state_dict()
        return {k: v.cpu().tolist() for k, v in state.items()}, _predict

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
