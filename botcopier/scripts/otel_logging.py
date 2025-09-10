import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id


class TraceContextFilter(logging.Filter):
    """Attach ``trace_id`` and ``span_id`` attributes to log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple
        span = trace.get_current_span()
        ctx = span.get_span_context()
        record.trace_id = format_trace_id(ctx.trace_id)
        record.span_id = format_span_id(ctx.span_id)
        return True


def setup_logging(service_name: str, level: int = logging.INFO) -> trace.Tracer:
    """Configure OpenTelemetry tracing and logging.

    Parameters
    ----------
    service_name:
        Name of the service emitting logs.
    level:
        Logging level to configure ``logging.basicConfig`` with.
    Returns
    -------
    ``Tracer`` instance for the service.
    """

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    try:  # prefer systemd journal if available
        from systemd.journal import JournalHandler
        logging.basicConfig(
            level=level,
            handlers=[JournalHandler()],
            format="%(asctime)s %(levelname)s [trace_id=%(trace_id)s span_id=%(span_id)s] %(message)s",
        )
    except Exception:  # pragma: no cover - fallback to file logging
        logging.basicConfig(
            level=level,
            filename=f"{service_name}.log",
            format="%(asctime)s %(levelname)s [trace_id=%(trace_id)s span_id=%(span_id)s] %(message)s",
        )
    logging.getLogger().addFilter(TraceContextFilter())
    return trace.get_tracer(service_name)
