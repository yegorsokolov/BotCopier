"""Feature engineering utilities for BotCopier."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING

from joblib import Memory, Parallel, delayed

try:  # optional numpy dependency
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - optional import
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # optional polars dependency
    import polars as pl  # type: ignore

    _HAS_POLARS = True
except Exception:  # pragma: no cover - optional import
    pl = None  # type: ignore
    _HAS_POLARS = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

from ..scripts.features import KALMAN_DEFAULT_PARAMS
from . import anomaly as _anomaly
from . import augmentation as _augmentation
from . import technical as _technical


@dataclass
class FeatureConfig:
    """Runtime configuration for feature engineering."""

    cache_dir: Path | str | None = None
    kalman_params: dict = field(default_factory=lambda: KALMAN_DEFAULT_PARAMS.copy())
    enabled_features: set[str] = field(default_factory=set)
    n_jobs: int | None = None
    memory: Memory = field(init=False, repr=False)
    feature_results: dict[int, tuple] = field(default_factory=dict, repr=False)
    feature_metadata_cache: dict[int, dict[str, dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )
    _cached_wrappers: dict[
        str, tuple[Callable[..., object], Callable[..., object]]
    ] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Initialise joblib memory using the current cache directory."""
        if hasattr(self, "memory"):
            try:
                self.memory.clear()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self.memory = Memory(
            str(self.cache_dir) if self.cache_dir else None, verbose=0
        )
        self._cached_wrappers.clear()

    def cache_function(
        self, name: str, func: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Return a joblib cached wrapper for ``func`` tied to this config."""

        entry = self._cached_wrappers.get(name)
        if entry is None or entry[0] is not func:
            cached_func = self.memory.cache(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                if cached_func.check_call_in_cache(*args, **kwargs):
                    logging.info("cache hit for %s", name)
                return cached_func(*args, **kwargs)

            wrapper.clear_cache = cached_func.clear  # type: ignore[attr-defined]
            entry = (func, wrapper)
            self._cached_wrappers[name] = entry
        return entry[1]

    def clear(self) -> None:
        """Clear cached feature state bound to this configuration."""

        self.memory.clear()
        self.feature_results.clear()
        self.feature_metadata_cache.clear()
        self._cached_wrappers.clear()

    def is_enabled(self, flag: str) -> bool:
        return flag in self.enabled_features

    @contextmanager
    def override(self, **kwargs: object):
        """Temporarily override configuration values."""

        old = {k: getattr(self, k) for k in kwargs}
        try:
            for k, v in kwargs.items():
                setattr(self, k, v)
            if "cache_dir" in kwargs:
                self.refresh()
            yield self
        finally:
            for k, v in old.items():
                setattr(self, k, v)
            if "cache_dir" in kwargs:
                self.refresh()


def _resolve_n_jobs(config: FeatureConfig, n_jobs: int | None) -> int:
    """Return an effective ``n_jobs`` respecting the provided configuration."""

    configured = config.n_jobs if config.n_jobs is not None else 1
    effective = n_jobs if n_jobs is not None else configured
    if not effective or effective <= 1:
        return 1
    return int(effective)


def _clone_frame(df):
    """Create a shallow copy of ``df`` preserving the backend."""

    clone = getattr(df, "clone", None)
    if callable(clone):  # polars offers ``clone``
        return clone()
    copy = getattr(df, "copy", None)
    if callable(copy):
        try:
            return copy(deep=False)
        except TypeError:  # pragma: no cover - some backends lack ``deep`` keyword
            return copy()
    return df


def _to_pandas_with_converter(df):
    """Return ``df`` as pandas along with a converter to the original type."""

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency missing
        raise ImportError("pandas is required for feature engineering") from exc

    if _HAS_POLARS and pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas(), pl.from_pandas  # type: ignore[arg-type]
    if isinstance(df, pd.DataFrame):
        return df, lambda data: data
    # Fallback: attempt to coerce via pandas constructor
    return pd.DataFrame(df), lambda data: data


def _merge_feature_frames(
    base_df,
    feature_names: Sequence[str],
    plugin_results: Sequence[tuple[str, object, Sequence[str]]],
):
    """Merge plugin outputs with ``base_df`` using vectorised dataframe ops."""

    if not plugin_results:
        return base_df, list(feature_names)

    import pandas as pd  # type: ignore

    base_pdf, to_original = _to_pandas_with_converter(base_df)
    plugin_frames, feature_arrays = zip(
        *(
            (
                _to_pandas_with_converter(plugin_df)[0],
                np.asarray(plugin_features, dtype=object)
                if _HAS_NUMPY
                else tuple(plugin_features),
            )
            for _, plugin_df, plugin_features in plugin_results
        )
    )

    frames: list[pd.DataFrame] = [base_pdf, *plugin_frames]
    merged_pdf = pd.concat(frames, axis=1, copy=False)
    merged_pdf = merged_pdf.loc[:, ~merged_pdf.columns.duplicated(keep="last")]

    if _HAS_NUMPY:
        combined = np.concatenate(
            [np.asarray(feature_names, dtype=object), *feature_arrays]
        )
        merged_features = (
            pd.Index(combined, dtype=object).drop_duplicates().tolist()
        )
    else:  # pragma: no cover - fallback without numpy
        merged_features = list(feature_names)
        for names in feature_arrays:
            for name in names:
                if name not in merged_features:
                    merged_features.append(name)

    return to_original(merged_pdf), merged_features


def _apply_parallel_plugins(
    df,
    feature_names: Sequence[str],
    plugin_names: Sequence[str],
    *,
    config: FeatureConfig,
    kwargs: dict,
    n_jobs: int | None = None,
    calendar_executed: bool = False,
):
    """Execute ``plugin_names`` possibly in parallel and merge their outputs."""

    if not plugin_names:
        return df, list(feature_names), {}, {}

    from .registry import FEATURE_REGISTRY

    effective_jobs = _resolve_n_jobs(config, n_jobs)
    tasks: list[tuple[str, object, dict]] = []
    for name in plugin_names:
        func = FEATURE_REGISTRY.get(name)
        if func is None:
            continue
        plugin_kwargs = dict(kwargs)
        if name == "technical" and calendar_executed:
            plugin_kwargs["calendar_features"] = False
        tasks.append((name, func, plugin_kwargs))

    if not tasks:
        return df, list(feature_names), {}, {}

    def _execute(task: tuple[str, object, dict]):
        name, func, plugin_kwargs = task
        local_df = _clone_frame(df)
        local_features = list(feature_names)
        result_df, result_features, emb, gnn = _call_plugin(
            func, config, local_df, local_features, plugin_kwargs
        )
        return name, result_df, result_features, emb or {}, gnn or {}

    if effective_jobs <= 1 or len(tasks) == 1:
        results = [_execute(task) for task in tasks]
    else:
        results = Parallel(n_jobs=effective_jobs, prefer="threads")(
            delayed(_execute)(task) for task in tasks
        )

    merged_df, merged_features = _merge_feature_frames(
        df,
        feature_names,
        [(name, res_df, res_features) for name, res_df, res_features, _, _ in results],
    )

    embeddings = {
        key: value
        for _, _, _, emb, _ in results
        for key, value in (emb or {}).items()
    }
    gnn_state = {
        key: value
        for _, _, _, _, gnn in results
        for key, value in (gnn or {}).items()
    }

    return merged_df, merged_features, embeddings, gnn_state


def _call_plugin(
    func: Callable[..., tuple],
    config: FeatureConfig,
    df,
    feature_names,
    plugin_kwargs: dict,
):
    try:
        return func(df, feature_names, config=config, **plugin_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "config" in message and "unexpected keyword argument" in message:
            return func(df, feature_names, **plugin_kwargs)
        raise


def configure_cache(config: FeatureConfig | None = None) -> FeatureConfig:
    """Prepare ``config`` for feature extraction and return it."""

    cfg = config or FeatureConfig()
    cfg.refresh()
    cfg.feature_results.clear()
    cfg.feature_metadata_cache.clear()

    from .registry import FEATURE_REGISTRY

    FEATURE_REGISTRY["augment_dataframe"] = _augmentation._augment_dataframe_impl
    FEATURE_REGISTRY["augment_dtw_dataframe"] = _augmentation._augment_dtw_dataframe_impl
    FEATURE_REGISTRY["technical"] = _technical._extract_features_impl
    _technical._FEATURE_METADATA.clear()
    return cfg


def clear_cache(config: FeatureConfig) -> None:
    """Remove cached feature computations bound to ``config``."""

    config.clear()
    _technical._FEATURE_METADATA.clear()


def _augment_dataframe(*args, config: FeatureConfig | None = None, **kwargs):
    cfg = config or FeatureConfig()
    func = cfg.cache_function("_augment_dataframe", _augmentation._augment_dataframe_impl)
    return func(*args, **kwargs)


def _augment_dtw_dataframe(*args, config: FeatureConfig | None = None, **kwargs):
    cfg = config or FeatureConfig()
    func = cfg.cache_function(
        "_augment_dtw_dataframe", _augmentation._augment_dtw_dataframe_impl
    )
    return func(*args, **kwargs)


def _extract_features(*args, **kwargs):
    config = kwargs.pop("config", None)
    cfg = config or FeatureConfig()
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("n_jobs", cfg.n_jobs)
    return _technical._extract_features(*args, config=cfg, **call_kwargs)


def _neutralize_against_market_index(*args, **kwargs):
    return _technical._neutralize_against_market_index(*args, **kwargs)


def _clip_train_features(*args, **kwargs):
    return _anomaly._clip_train_features(*args, **kwargs)


def _clip_apply(*args, **kwargs):
    return _anomaly._clip_apply(*args, **kwargs)


def _score_anomalies(*args, **kwargs):
    return _anomaly._score_anomalies(*args, **kwargs)


def train(*args, **kwargs):
    from botcopier.training.pipeline import train as _train

    return _train(*args, **kwargs)
