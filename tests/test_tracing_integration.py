import json
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp_module

from logging_utils import setup_logging
from botcopier.training.pipeline import train

def _dummy_exporter_factory(exporters):
    class DummyExporter(InMemorySpanExporter):
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__()
            exporters.append(self)
    return DummyExporter

def test_tracing_spans_emitted(tmp_path, monkeypatch):
    exporters: list[InMemorySpanExporter] = []
    monkeypatch.setattr(otlp_module, "OTLPSpanExporter", _dummy_exporter_factory(exporters))

    setup_logging(enable_tracing=True, exporter="otlp")

    data = tmp_path / "trades_raw.csv"
    data.write_text(
        "label,price,volume,spread,hour,symbol\n"
        "0,1.0,100,1.5,0,EURUSD\n"
        "1,1.1,110,1.6,1,EURUSD\n"
        "0,1.2,120,1.7,2,EURUSD\n"
        "1,1.3,130,1.8,3,EURUSD\n"
    )
    out_dir = tmp_path / "out"
    train(data, out_dir)

    trace.get_tracer_provider().force_flush()  # ensure spans exported
    assert exporters, "Exporter should be instantiated"
    spans = exporters[0].get_finished_spans()
    names = {s.name for s in spans}
    assert {"data_load", "feature_extraction", "model_fit", "evaluation"} <= names
