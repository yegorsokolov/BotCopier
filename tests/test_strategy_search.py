import json
import sys
import types
from pathlib import Path

import numpy as np

# Stub heavy optional dependencies so pipeline imports without installing them
pandas_stub = types.ModuleType("pandas")
pandas_stub.DataFrame = type("DataFrame", (), {})
sys.modules.setdefault("pandas", pandas_stub)
sys.modules.setdefault("psutil", types.ModuleType("psutil"))

scipy = types.ModuleType("scipy")
scipy.cluster = types.ModuleType("cluster")
scipy.cluster.hierarchy = types.ModuleType("hierarchy")
scipy.spatial = types.ModuleType("spatial")
scipy.spatial.distance = types.ModuleType("distance")
scipy.cluster.hierarchy.fcluster = lambda *a, **k: None
scipy.cluster.hierarchy.linkage = lambda *a, **k: None
scipy.cluster.hierarchy.leaves_list = lambda *a, **k: []
scipy.spatial.distance.squareform = lambda *a, **k: None
sys.modules.update(
    {
        "scipy": scipy,
        "scipy.cluster": scipy.cluster,
        "scipy.cluster.hierarchy": scipy.cluster.hierarchy,
        "scipy.spatial": scipy.spatial,
        "scipy.spatial.distance": scipy.spatial.distance,
    }
)

sklearn = types.ModuleType("sklearn")
sklearn.calibration = types.ModuleType("calibration")
sklearn.feature_selection = types.ModuleType("feature_selection")
sklearn.preprocessing = types.ModuleType("preprocessing")
sklearn.calibration.CalibratedClassifierCV = object
sklearn.calibration.calibration_curve = lambda *a, **k: ([], [])
sklearn.feature_selection.mutual_info_classif = lambda *a, **k: None
sklearn.preprocessing.PowerTransformer = object
sklearn.preprocessing.StandardScaler = object
sklearn.preprocessing.RobustScaler = object
sklearn.ensemble = types.ModuleType("ensemble")
sklearn.ensemble.IsolationForest = object
sklearn.base = types.ModuleType("base")
sklearn.base.BaseEstimator = type("BaseEstimator", (), {})
sklearn.base.TransformerMixin = type("TransformerMixin", (), {})
sklearn.linear_model = types.ModuleType("linear_model")
sklearn.linear_model.LogisticRegression = object
sklearn.pipeline = types.ModuleType("pipeline")
sklearn.pipeline.Pipeline = object
sklearn.metrics = types.ModuleType("metrics")
for _m in [
    "accuracy_score",
    "average_precision_score",
    "brier_score_loss",
    "roc_auc_score",
]:
    setattr(sklearn.metrics, _m, lambda *a, **k: 0.0)
sys.modules.update(
    {
        "sklearn": sklearn,
        "sklearn.calibration": sklearn.calibration,
        "sklearn.feature_selection": sklearn.feature_selection,
        "sklearn.preprocessing": sklearn.preprocessing,
        "sklearn.ensemble": sklearn.ensemble,
        "sklearn.base": sklearn.base,
        "sklearn.linear_model": sklearn.linear_model,
        "sklearn.pipeline": sklearn.pipeline,
        "sklearn.metrics": sklearn.metrics,
    }
)

opentelemetry = types.ModuleType("opentelemetry")
opentelemetry.trace = types.ModuleType("trace")
opentelemetry.trace.get_tracer = lambda *a, **k: types.SimpleNamespace(
    start_as_current_span=lambda *a, **k: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc_type, exc, tb: None)
)
sys.modules.update({"opentelemetry": opentelemetry, "opentelemetry.trace": opentelemetry.trace})

pydantic = types.ModuleType("pydantic")
pydantic.ValidationError = type("ValidationError", (Exception,), {})
class _BaseModel:
    model_fields = {"version": types.SimpleNamespace(default=1)}

    def __init__(self, **kwargs):
        pass

    def model_dump(self, *a, **k):
        return {}

pydantic.BaseModel = _BaseModel
pydantic.ConfigDict = dict
pydantic.Field = lambda *a, **k: None
sys.modules["pydantic"] = pydantic
features_pkg = types.ModuleType("botcopier.features")
features_pkg.__path__ = []  # mark as package

technical_mod = types.ModuleType("technical")
technical_mod._extract_features = lambda df, names, n_jobs=None: (df, names, None, None)
technical_mod._neutralize_against_market_index = lambda df, n_jobs=None: df

anomaly_mod = types.ModuleType("anomaly")
anomaly_mod._clip_train_features = lambda df, names=None: (df, names)

engineering_mod = types.ModuleType("engineering")

class FeatureConfig:
    def __init__(self, cache_dir=None, enabled_features=None):
        self.cache_dir = cache_dir
        self.enabled_features = enabled_features

engineering_mod.FeatureConfig = FeatureConfig
engineering_mod.configure_cache = lambda config: None

sys.modules.update(
    {
        "botcopier.features": features_pkg,
        "botcopier.features.technical": technical_mod,
        "botcopier.features.anomaly": anomaly_mod,
        "botcopier.features.engineering": engineering_mod,
        "botcopier.features.augmentation": types.SimpleNamespace(
            _augment_dataframe=lambda df, *a, **k: df,
            _augment_dtw_dataframe=lambda df, *a, **k: df,
        ),
    }
)

yaml = types.ModuleType("yaml")
yaml.safe_load = lambda *a, **k: {}
yaml.safe_dump = lambda *a, **k: ""
sys.modules["yaml"] = yaml

pydantic_settings = types.ModuleType("pydantic_settings")

class _BaseSettings:
    def model_copy(self, update=None):
        return self

    def model_dump(self):
        return {}

pydantic_settings.BaseSettings = _BaseSettings
sys.modules["pydantic_settings"] = pydantic_settings

pandera = types.ModuleType("pandera")
pandera.Field = lambda *a, **k: None
pandera.DataFrameModel = type("DataFrameModel", (), {})
pandera.typing = types.ModuleType("typing")
pandera.typing.Series = object
sys.modules.update({"pandera": pandera, "pandera.typing": pandera.typing})

jinja2 = types.ModuleType("jinja2")
jinja2.Environment = object
jinja2.select_autoescape = lambda *a, **k: None
sys.modules["jinja2"] = jinja2

from botcopier.strategy import (
    Price,
    SMA,
    GT,
    Position,
    backtest,
    deserialize,
    serialize,
    search_strategy,
)
from botcopier.training.pipeline import train


def test_strategy_search_outperforms_baseline(tmp_path: Path) -> None:
    prices = np.linspace(1.0, 100.0, 200)

    baseline = Position(GT(Price(), Price()))
    baseline_ret = backtest(prices, baseline)

    best, score = search_strategy(prices, n_samples=25)
    assert score > baseline_ret

    compiled = deserialize(serialize(best))
    assert np.allclose(backtest(prices, compiled), backtest(prices, best))

    out_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    out_dir.mkdir()
    data_dir.mkdir()
    train(data_dir, out_dir, strategy_search=True)

    model = json.loads((out_dir / "model.json").read_text())
    assert "strategy" in model and "strategy_score" in model
    expr = deserialize(model["strategy"])
    assert backtest(prices, expr) >= baseline_ret
