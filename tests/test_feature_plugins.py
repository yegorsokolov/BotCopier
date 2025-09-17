import types
import sys

import pandas as pd
from importlib.metadata import EntryPoint

# stub minimal sklearn module to avoid heavy dependency when unavailable
try:
    import sklearn  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    sklearn = types.ModuleType("sklearn")
    sklearn.ensemble = types.ModuleType("sklearn.ensemble")
    sklearn.ensemble.IsolationForest = object
    sklearn.linear_model = types.ModuleType("sklearn.linear_model")
    sklearn.linear_model.LinearRegression = object
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.ensemble", sklearn.ensemble)
    sys.modules.setdefault("sklearn.linear_model", sklearn.linear_model)

# stub minimal scipy module when not installed
try:
    import scipy  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    scipy = types.ModuleType("scipy")
    scipy.signal = types.ModuleType("scipy.signal")
    sys.modules.setdefault("scipy", scipy)
    sys.modules.setdefault("scipy.signal", scipy.signal)

# stub gplearn
gplearn = types.ModuleType("gplearn")
gplearn.genetic = types.ModuleType("gplearn.genetic")
gplearn.genetic.SymbolicTransformer = object
sys.modules.setdefault("gplearn", gplearn)
sys.modules.setdefault("gplearn.genetic", gplearn.genetic)

# stub psutil
sys.modules.setdefault("psutil", types.ModuleType("psutil"))

# stub joblib Memory
joblib = types.ModuleType("joblib")

class _DummyMemory:
    def __init__(self, *args, **kwargs):
        pass

    def cache(self, func):
        def wrapper(*a, **kw):
            return func(*a, **kw)

        wrapper.clear = lambda: None
        wrapper.check_call_in_cache = lambda *a, **kw: False
        return wrapper

    def clear(self):
        pass

joblib.Memory = _DummyMemory
class _DummyParallel:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, iterable):
        return [func() for func in iterable]


def _dummy_delayed(func):
    def wrapper(*args, **kwargs):
        return lambda: func(*args, **kwargs)

    return wrapper


joblib.Parallel = _DummyParallel
joblib.delayed = _dummy_delayed
sys.modules.setdefault("joblib", joblib)

from botcopier.features.engineering import FeatureConfig, configure_cache
import botcopier.features.registry as registry
from botcopier.features.registry import FEATURE_REGISTRY
from botcopier.features import technical


def test_entry_point_plugin(monkeypatch):
    # create a fake third-party module with a feature function
    mod = types.ModuleType("thirdparty_mod")

    def feature(df, names, **kwargs):
        df["ep"] = 1.0
        names.append("ep")
        return df, names, {}, {}

    mod.feature = feature
    sys.modules["thirdparty_mod"] = mod

    # mock entry points to advertise the plugin
    ep = EntryPoint(name="ep", value="thirdparty_mod:feature", group="botcopier.features")
    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda group=None: [ep] if group == "botcopier.features" else [],
    )

    # enable plugin via configuration
    configure_cache(FeatureConfig(enabled_features={"ep"}))

    # replace heavy technical plugin with a no-op for the test
    FEATURE_REGISTRY["technical"] = lambda df, names, **kwargs: (df, names, {}, {})

    df = pd.DataFrame({"price": [1.0]})
    out, cols, *_ = technical._extract_features(df, [])
    assert "ep" in cols
    assert out["ep"].iloc[0] == 1.0


def test_register_feature_direct():
    """Plugins can be registered directly via ``register_feature``."""

    @registry.register_feature("direct")
    def direct_feature(df, names, **kwargs):
        df["direct"] = 2.0
        names.append("direct")
        return df, names, {}, {}

    configure_cache(FeatureConfig(enabled_features={"direct"}))

    # replace heavy technical plugin with a no-op for the test
    FEATURE_REGISTRY["technical"] = lambda df, names, **kwargs: (df, names, {}, {})

    df = pd.DataFrame({"price": [1.0]})
    out, cols, *_ = technical._extract_features(df, [])
    assert "direct" in cols
    assert out["direct"].iloc[0] == 2.0
    FEATURE_REGISTRY.pop("direct", None)


def test_builtin_plugins_registered():
    """Core feature plugins are exposed through the registry."""

    registry.load_plugins()
    available = set(FEATURE_REGISTRY)
    expected = {
        "lag_diff",
        "technical_indicators",
        "wavelet_packets",
        "rolling_correlations",
        "graph_embeddings",
    }
    assert expected.issubset(available)
