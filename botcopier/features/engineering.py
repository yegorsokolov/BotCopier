"""Feature engineering utilities for BotCopier."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

from joblib import Memory

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

    def is_enabled(self, flag: str) -> bool:
        return flag in self.enabled_features

    @contextmanager
    def override(self, **kwargs: object):
        """Temporarily override configuration values."""
        old = {k: getattr(self, k) for k in kwargs}
        try:
            for k, v in kwargs.items():
                setattr(self, k, v)
            configure_cache(self)
            yield self
        finally:
            for k, v in old.items():
                setattr(self, k, v)
            configure_cache(self)


_CONFIG = FeatureConfig()
_MEMORY = Memory(None, verbose=0)
_FEATURE_RESULTS: dict[int, tuple] = {}


def configure_cache(config: FeatureConfig) -> None:
    """Configure joblib cache directory for expensive feature functions."""
    global _MEMORY, _CONFIG
    _CONFIG = config
    _MEMORY = Memory(str(config.cache_dir) if config.cache_dir else None, verbose=0)

    from .registry import FEATURE_REGISTRY

    _augmentation._augment_dataframe = _cache_with_logging(
        _augmentation._augment_dataframe_impl, "_augment_dataframe"
    )
    FEATURE_REGISTRY["augment_dataframe"] = _augmentation._augment_dataframe
    _augmentation._augment_dtw_dataframe = _cache_with_logging(
        _augmentation._augment_dtw_dataframe_impl, "_augment_dtw_dataframe"
    )
    FEATURE_REGISTRY["augment_dtw_dataframe"] = _augmentation._augment_dtw_dataframe
    _technical._csd_pair = _cache_with_logging(_technical._csd_pair_impl, "_csd_pair")
    FEATURE_REGISTRY["technical"] = _technical._extract_features_impl


def clear_cache() -> None:
    """Remove all cached feature computations."""
    _MEMORY.clear()
    _FEATURE_RESULTS.clear()


def _cache_with_logging(func, name: str):
    cached_func = _MEMORY.cache(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if cached_func.check_call_in_cache(*args, **kwargs):
            logging.info("cache hit for %s", name)
        return cached_func(*args, **kwargs)

    wrapper.clear_cache = cached_func.clear  # type: ignore[attr-defined]
    return wrapper


def _augment_dataframe(*args, **kwargs):
    return _augmentation._augment_dataframe(*args, **kwargs)


def _augment_dtw_dataframe(*args, **kwargs):
    return _augmentation._augment_dtw_dataframe(*args, **kwargs)


def _extract_features(*args, **kwargs):
    return _technical._extract_features(*args, **kwargs)


def _neutralize_against_market_index(*args, **kwargs):
    return _technical._neutralize_against_market_index(*args, **kwargs)


def _clip_train_features(*args, **kwargs):
    return _anomaly._clip_train_features(*args, **kwargs)


def _clip_apply(*args, **kwargs):
    return _anomaly._clip_apply(*args, **kwargs)


def _score_anomalies(*args, **kwargs):
    return _anomaly._score_anomalies(*args, **kwargs)


configure_cache(_CONFIG)


def train(*args, **kwargs):
    from botcopier.training.pipeline import train as _train

    return _train(*args, **kwargs)
