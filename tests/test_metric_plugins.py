import sys
import types

from importlib.metadata import EntryPoint

# stub optional dependencies required by evaluation import
pyarrow = types.ModuleType("pyarrow")


class _ArrowField:
    def __init__(self, name: str):
        self.name = name


def _make_type():  # pragma: no cover - trivial stub
    return object()


pyarrow.int32 = _make_type  # type: ignore[attr-defined]
pyarrow.float64 = _make_type  # type: ignore[attr-defined]
pyarrow.string = _make_type  # type: ignore[attr-defined]


def _schema(fields):  # pragma: no cover - trivial stub
    return [_ArrowField(name) for name, *_ in fields]


pyarrow.schema = _schema  # type: ignore[attr-defined]
pyarrow.Schema = list  # type: ignore[attr-defined]
sys.modules.setdefault("pyarrow", pyarrow)
pydantic = types.ModuleType("pydantic")


class _BaseModel:  # pragma: no cover - minimal stub
    def model_dump(self, *args, **kwargs):
        return {}


pydantic.BaseModel = _BaseModel  # type: ignore[attr-defined]
sys.modules.setdefault("pydantic", pydantic)

from botcopier.scripts import evaluation
import metrics.registry as metric_registry


def test_entry_point_metric(monkeypatch):
    original = metric_registry._REGISTRY.copy()
    try:
        module = types.ModuleType("thirdparty_metric_mod")

        def metric(y_true, probas, profits=None):  # type: ignore[unused-arg]
            return "plugin"

        module.metric = metric
        sys.modules["thirdparty_metric_mod"] = module

        ep = EntryPoint(
            name="plugin_metric",
            value="thirdparty_metric_mod:metric",
            group="botcopier.metrics",
        )
        monkeypatch.setattr(
            metric_registry,
            "entry_points",
            lambda group=None: [ep] if group == "botcopier.metrics" else [],
        )

        result = evaluation._classification_metrics(
            [], [], None, selected=["plugin_metric"]
        )
        assert result["plugin_metric"] == "plugin"
    finally:
        metric_registry._REGISTRY = original
        sys.modules.pop("thirdparty_metric_mod", None)
