import os
import logging
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

# Metric registry: name -> (type, unit, description)
METRICS_REGISTRY = {
    "mcp.tool_calls.count": ("counter", "1", "Tool call invocations"),
    "mcp.tool_calls.errors": ("counter", "1", "Tool call errors"),
    "mcp.search.count": ("counter", "1", "Search invocations"),
    "mcp.cache.writes": ("counter", "1", "Cache chunk writes"),
    "mcp.cache.reads": ("counter", "1", "Cache chunk reads"),
    "mcp.sync.runs": ("counter", "1", "Sync job runs"),
    "mcp.tool_calls.latency_ms": ("histogram", "ms", "Downstream MCP call latency"),
    "mcp.search.latency_ms": ("histogram", "ms", "find_tools end-to-end latency"),
    "mcp.embedding.latency_ms": ("histogram", "ms", "vLLM embedding latency"),
    "mcp.summary.latency_ms": ("histogram", "ms", "LLM summary latency"),
    "mcp.cache.write_latency_ms": ("histogram", "ms", "Async cache write latency"),
    "mcp.tools.total": ("gauge", "1", "Current tool count"),
    "mcp.cache.queue_depth": ("gauge", "1", "Pending cache writes"),
    "mcp.sessions.active": ("gauge", "1", "Active session count"),
}

_meter: metrics.Meter | None = None
_metric_instruments: dict = {}
_tracer: trace.Tracer | None = None


def setup_telemetry(settings) -> None:
    """Initialize Phoenix Arize OTEL. Falls back gracefully if disabled."""
    global _meter, _tracer

    if not settings.otel.enabled:
        logger.info("Telemetry disabled")
        _meter = metrics.get_meter(__name__)
        _tracer = trace.get_tracer(__name__)
        _register_metrics()
        return

    # Set Phoenix env vars before importing phoenix
    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", settings.otel.endpoint)
    if settings.otel.api_key:
        os.environ.setdefault("PHOENIX_API_KEY", settings.otel.api_key)

    try:
        from phoenix.otel import register
        register(
            project_name=settings.otel.project_name,
            auto_instrument=True,
        )
        logger.info("Phoenix OTEL registered successfully")
    except Exception as exc:
        logger.warning("Phoenix OTEL registration failed (continuing): %s", exc)

    _meter = metrics.get_meter(__name__)
    _tracer = trace.get_tracer(__name__)
    _register_metrics()


def _register_metrics() -> None:
    global _metric_instruments
    if _meter is None:
        return
    for name, (kind, unit, desc) in METRICS_REGISTRY.items():
        safe_name = name.replace(".", "_")
        if kind == "counter":
            _metric_instruments[name] = _meter.create_counter(name, unit=unit, description=desc)
        elif kind == "histogram":
            _metric_instruments[name] = _meter.create_histogram(name, unit=unit, description=desc)
        elif kind == "gauge":
            _metric_instruments[name] = _meter.create_gauge(name, unit=unit, description=desc)


def get_tracer() -> trace.Tracer:
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(__name__)
    return _tracer


def record_counter(name: str, value: int = 1, attributes: dict | None = None) -> None:
    inst = _metric_instruments.get(name)
    if inst is not None:
        inst.add(value, attributes or {})


def record_histogram(name: str, value: float, attributes: dict | None = None) -> None:
    inst = _metric_instruments.get(name)
    if inst is not None:
        inst.record(value, attributes or {})


def record_gauge(name: str, value: float, attributes: dict | None = None) -> None:
    inst = _metric_instruments.get(name)
    if inst is not None:
        inst.set(value, attributes or {})


def get_metric_instruments() -> dict:
    return _metric_instruments
