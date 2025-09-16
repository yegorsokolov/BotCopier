import json
import logging
import os
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Serialise log records to JSON with a common schema."""

    #: Fields added by :class:`logging.LogRecord` that should be ignored when
    #: merging ``extra`` attributes into the JSON payload.  The list mirrors the
    #: attributes documented in :mod:`logging` to avoid leaking internal state.
    _RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - thin
        message = record.getMessage()
        log: dict[str, object] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
        }

        if isinstance(record.msg, dict):
            log.update(self._coerce_dict(record.msg))
        elif message:
            log["message"] = message

        # Attach structured context from ``extra`` parameters.
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._RESERVED and not key.startswith("_") and value is not None
        }
        if extras:
            log.update(self._coerce_dict(extras))

        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)

        # Include trace context when present.  OpenTelemetry injects the IDs as
        # ``otelTraceID`` and ``otelSpanID`` attributes.
        trace_id = getattr(record, "otelTraceID", None)
        span_id = getattr(record, "otelSpanID", None)
        if trace_id:
            log["trace_id"] = trace_id
        if span_id:
            log["span_id"] = span_id

        return json.dumps(log, default=self._coerce)

    def _coerce(self, value: object) -> object:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple, set)):
            return [self._coerce(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._coerce(v) for k, v in value.items()}
        return str(value)

    def _coerce_dict(self, data: dict[str, object]) -> dict[str, object]:
        return {key: self._coerce(val) for key, val in data.items()}


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

    env_level = os.getenv("BOTCOPIER_LOG_LEVEL")
    if env_level:
        try:
            level = int(env_level)
        except ValueError:
            level_name = env_level.upper()
            level = getattr(logging, level_name, level)

    structured_flag = os.getenv("BOTCOPIER_LOG_FORMAT")
    structured = True
    if structured_flag:
        structured = structured_flag.lower() not in {"plain", "text", "human"}
    elif (env_structured := os.getenv("BOTCOPIER_JSON_LOGS")) is not None:
        structured = env_structured.lower() in {"1", "true", "yes", "on"}

    root = logging.getLogger()
    if not getattr(root, "_botcopier_configured", False):
        root.handlers.clear()
        root._botcopier_configured = True  # type: ignore[attr-defined]

    handler: logging.Handler | None = None
    for existing in root.handlers:
        if isinstance(existing, logging.StreamHandler):
            handler = existing
            break
    if handler is None:
        handler = logging.StreamHandler()
        root.addHandler(handler)

    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        fmt = os.getenv(
            "BOTCOPIER_LOG_FMT",
            "%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        datefmt = os.getenv("BOTCOPIER_LOG_DATEFMT", "%Y-%m-%dT%H:%M:%S%z")
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.setLevel(level)

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

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
