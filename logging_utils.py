import json
import logging
import os
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple
        log = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)
        # Include trace context if present
        trace_id = getattr(record, "otelTraceID", None)
        span_id = getattr(record, "otelSpanID", None)
        if trace_id and span_id:
            log["trace_id"] = trace_id
            log["span_id"] = span_id
        return json.dumps(log)


def setup_logging(
    name: Optional[str] = None,
    level: int = logging.INFO,
    enable_tracing: bool = False,
    exporter: str = "otlp",
) -> logging.Logger:
    """Configure basic JSON logging and optionally OpenTelemetry tracing.

    Parameters
    ----------
    name:
        Name of the logger to return. ``None`` returns the root logger.
    level:
        Logging level for the returned logger.
    enable_tracing:
        If ``True``, initialise an OpenTelemetry tracer.
    exporter:
        Exporter to use when tracing is enabled. ``"otlp"`` or ``"jaeger"``.
    """

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    if enable_tracing:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            if exporter == "jaeger":
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                jaeger_host = os.getenv("JAEGER_HOST", "localhost")
                jaeger_port = int(os.getenv("JAEGER_PORT", "6831"))
                span_exporter = JaegerExporter(
                    agent_host_name=jaeger_host, agent_port=jaeger_port
                )
            else:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                endpoint = os.getenv(
                    "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"
                )
                span_exporter = OTLPSpanExporter(endpoint=endpoint)

            resource = Resource.create({"service.name": name or "botcopier"})
            current_provider = trace.get_tracer_provider()
            if isinstance(current_provider, TracerProvider):
                provider = current_provider
            else:
                provider = TracerProvider(resource=resource)
                trace.set_tracer_provider(provider)
            provider.add_span_processor(BatchSpanProcessor(span_exporter))
            try:
                from opentelemetry.instrumentation.logging import LoggingInstrumentor

                LoggingInstrumentor().instrument()
            except Exception:  # pragma: no cover - optional
                pass
        except Exception:  # pragma: no cover - optional
            pass

    return logging.getLogger(name)
