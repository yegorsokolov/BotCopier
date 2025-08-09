#!/usr/bin/env python3
"""Pre-train simple autoencoder on tick price changes."""
import argparse
import json
import csv
from pathlib import Path
from typing import List

import logging
import os
import numpy as np

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
try:  # Optional Jaeger exporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    JaegerExporter = None
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_trace_id, format_span_id

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    HAS_TF = True
except Exception:
    HAS_TF = False

from sklearn.cluster import KMeans


resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "pretrain_autoencoder")})
provider = TracerProvider(resource=resource)
if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
elif os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST") and JaegerExporter:
    provider.add_span_processor(
        BatchSpanProcessor(
            JaegerExporter(
                agent_host_name=os.getenv("OTEL_EXPORTER_JAEGER_AGENT_HOST"),
                agent_port=int(os.getenv("OTEL_EXPORTER_JAEGER_AGENT_PORT", "6831")),
            )
        )
    )
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

logger_provider = LoggerProvider(resource=resource)
if endpoint:
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint)))
set_logger_provider(logger_provider)
handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {"level": record.levelname}
        if isinstance(record.msg, dict):
            log.update(record.msg)
        else:
            log["message"] = record.getMessage()
        if hasattr(record, "trace_id"):
            log["trace_id"] = format_trace_id(record.trace_id)
        if hasattr(record, "span_id"):
            log["span_id"] = format_span_id(record.span_id)
        return json.dumps(log)


logger = logging.getLogger(__name__)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _load_ticks(file: Path) -> List[float]:
    """Load bid prices from TickHistoryExporter CSV."""
    prices: List[float] = []
    with open(file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            try:
                prices.append(float(r.get("bid", 0) or 0))
            except Exception:
                continue
    return prices


def train(
    tick_dir: Path,
    out_path: Path,
    *,
    window: int = 10,
    latent_dim: int = 4,
    epochs: int = 10,
    n_clusters: int = 4,
) -> None:
    if not HAS_TF:
        raise ImportError("TensorFlow is required for autoencoder training")
    with tracer.start_as_current_span("pretrain_autoencoder_train"):
        logger.info("training autoencoder")
        seqs: List[np.ndarray] = []
        for tick_file in tick_dir.glob("ticks_*.csv"):
            prices = _load_ticks(tick_file)
            if len(prices) <= window:
                continue
            diffs = np.diff(prices)
            for i in range(len(diffs) - window):
                seqs.append(diffs[i : i + window])
        if not seqs:
            raise ValueError(f"No tick sequences found in {tick_dir}")

        X = np.stack(seqs).astype(np.float32)

        inp = keras.Input(shape=(window,))
        x = keras.layers.Dense(latent_dim, use_bias=False)(inp)
        out = keras.layers.Dense(window, use_bias=False)(x)
        model = keras.Model(inp, out)
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, X, epochs=epochs, batch_size=32, verbose=0)

        encoder_weights = model.layers[1].get_weights()[0]  # type: ignore[index]

        encoded = X.dot(encoder_weights.astype(np.float32))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        km.fit(encoded)

        data = {
            "window": window,
            "weights": encoder_weights.tolist(),
            "centers": km.cluster_centers_.tolist(),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("encoder weights written", extra={"out_file": str(out_path)})


def main() -> None:
    with tracer.start_as_current_span("pretrain_autoencoder"):
        logger.info("pretrain autoencoder start")
        p = argparse.ArgumentParser(description="Pretrain autoencoder from ticks")
        p.add_argument("tick_dir", help="directory with ticks_*.csv files")
        p.add_argument("out_file", help="output JSON file for encoder weights")
        p.add_argument("--window", type=int, default=10)
        p.add_argument("--latent-dim", type=int, default=4)
        p.add_argument("--epochs", type=int, default=10)
        p.add_argument("--clusters", type=int, default=4)
        args = p.parse_args()
        train(
            Path(args.tick_dir),
            Path(args.out_file),
            window=args.window,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            n_clusters=args.clusters,
        )
        logger.info("pretrain autoencoder finished")


if __name__ == "__main__":
    main()
