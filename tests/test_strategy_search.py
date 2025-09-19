import json
import sys
import types
from pathlib import Path

import numpy as np


# Stub heavy optional dependencies so pipeline imports without installing them
try:  # pragma: no cover - executed only when pandas is unavailable
    import pandas  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback for minimal environments
    class _StubDataFrame:
        def select_dtypes(self, include=None):
            return self

        @property
        def empty(self) -> bool:
            return True

    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = _StubDataFrame
    pandas_stub.RangeIndex = type("RangeIndex", (), {})
    pandas_stub.read_csv = lambda *a, **k: _StubDataFrame()
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
sklearn.ensemble.GradientBoostingClassifier = object
sklearn.ensemble.VotingClassifier = object
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
engineering_mod.configure_cache = lambda config: config
engineering_mod._extract_features = (
    lambda df, names, n_jobs=None, **kwargs: (df, names, None, None)
)
engineering_mod._neutralize_against_market_index = lambda df, n_jobs=None: df

augmentation_mod = types.ModuleType("augmentation")
augmentation_mod._augment_dataframe = lambda df, *a, **k: df
augmentation_mod._augment_dtw_dataframe = lambda df, *a, **k: df

sys.modules.update(
    {
        "botcopier.features": features_pkg,
        "botcopier.features.technical": technical_mod,
        "botcopier.features.anomaly": anomaly_mod,
        "botcopier.features.engineering": engineering_mod,
        "botcopier.features.augmentation": augmentation_mod,
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

pyarrow_stub = types.ModuleType("pyarrow")
pyarrow_stub.schema = lambda *a, **k: None
pyarrow_stub.field = lambda *a, **k: None
pyarrow_stub.int32 = lambda *a, **k: None
pyarrow_stub.float64 = lambda *a, **k: None
pyarrow_stub.string = lambda *a, **k: None
sys.modules.setdefault("pyarrow", pyarrow_stub)

joblib_stub = types.ModuleType("joblib")


class _StubMemory:
    def __init__(self, *args, **kwargs):
        pass

    def cache(self, func):
        return func


joblib_stub.Memory = _StubMemory
sys.modules.setdefault("joblib", joblib_stub)

from botcopier.strategy import (
    ATR,
    BollingerBand,
    CrossPrice,
    GT,
    Position,
    Price,
    RSI,
    RollingVolatility,
    backtest,
    deserialize,
    search_strategies,
    serialize,
)
from botcopier.strategy.search import PARETO_MAX_SIZE
from botcopier.training.pipeline import train


def test_strategy_search_outperforms_baseline(tmp_path: Path) -> None:
    base = np.linspace(1.0, 100.0, 200)
    peer = base[::-1] + 5.0
    context = {"base": base, "peer": peer}

    baseline = Position(GT(Price(), Price()))
    baseline_ret = backtest(context, baseline)

    best, pareto = search_strategies(context, n_samples=25)
    assert best.ret >= baseline_ret
    assert best.complexity >= 1
    assert len(pareto) > 1
    assert any(
        any(
            isinstance(node, (RSI, ATR, BollingerBand, RollingVolatility, CrossPrice))
            for node in cand.expr.iter_nodes()
        )
        for cand in pareto
    )

    compiled = deserialize(serialize(best.expr))
    assert np.allclose(backtest(context, compiled), backtest(context, best.expr))

    baseline_array_ret = backtest(base, baseline)

    out_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    out_dir.mkdir()
    data_dir.mkdir()
    train(data_dir, out_dir, strategy_search=True)

    model = json.loads((out_dir / "model.json").read_text())
    assert "strategies" in model and model["strategies"]
    assert len(model["strategies"]) <= PARETO_MAX_SIZE
    first = model["strategies"][0]
    assert "complexity" in first
    expr = deserialize(first["expr"])
    assert backtest(base, expr) >= baseline_array_ret
    assert model.get("best_complexity", 0) >= 1
    metadata = model.get("strategy_search_metadata")
    assert metadata and metadata["population_size"] >= 12
    assert metadata["price_history"]["length"] >= 2
    assert metadata.get("use_curriculum") is True
    assert metadata.get("stage_count", 0) >= 1
    assert metadata.get("curriculum")
    first_stage = metadata["curriculum"][0]
    assert first_stage["start_generation"] == 0
    assert first_stage["max_depth"] >= 1
    assert metadata["final_max_depth"] >= first_stage["max_depth"]


def test_curriculum_scheduler_records_progress() -> None:
    base = np.linspace(1.0, 250.0, 256) + np.sin(np.linspace(0.0, 6.0, 256)) * 4.0
    peer = base[::-1] * 0.6 + 12.0
    context = {"base": base, "peer": peer}

    result = search_strategies(
        context,
        seed=7,
        population_size=18,
        n_generations=8,
        n_samples=144,
        use_curriculum=True,
    )
    metadata = result.metadata
    stages = metadata.get("curriculum", [])
    assert stages and metadata.get("use_curriculum") is True
    assert metadata.get("stage_count") == len(stages)
    assert stages[0]["start_generation"] == 0
    assert stages[-1]["end_generation"] is not None
    assert stages[-1]["max_depth"] == metadata["final_max_depth"]
    assert any(stage.get("advance_reason") == "score" for stage in stages if stage.get("advance_reason"))
    assert any(stage.get("score_threshold") is not None for stage in stages)


def test_curriculum_outperforms_static_depth() -> None:
    base = np.linspace(1.0, 320.0, 256) + np.cos(np.linspace(0.0, 5.0, 256)) * 6.0
    peer = np.roll(base, 3) * 0.8 + 8.0
    context = {"base": base, "peer": peer}

    curriculum = search_strategies(
        context,
        seed=8,
        population_size=20,
        n_generations=8,
        n_samples=160,
        max_depth=4,
        use_curriculum=True,
    )
    static = search_strategies(
        context,
        seed=8,
        population_size=20,
        n_generations=8,
        n_samples=160,
        max_depth=2,
        use_curriculum=False,
    )

    assert curriculum.best.ret > static.best.ret
    assert curriculum.best.complexity > static.best.complexity
    assert curriculum.metadata.get("curriculum")
    assert static.metadata.get("curriculum") == []
