"""GET /v1/metrics — Prometheus-compatible metrics endpoint."""
import logging
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from mcp_tool_manager.telemetry import get_metric_instruments, METRICS_REGISTRY

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Return Prometheus-compatible text format metrics."""
    lines = []
    instruments = get_metric_instruments()

    for name, (kind, unit, desc) in METRICS_REGISTRY.items():
        prom_name = name.replace(".", "_").replace("-", "_")
        lines.append(f"# HELP {prom_name} {desc}")
        lines.append(f"# TYPE {prom_name} {_prom_type(kind)}")
        # Note: actual values require OTEL metric reader; emit 0 as placeholder
        lines.append(f"{prom_name} 0")

    return "\n".join(lines) + "\n"


def _prom_type(kind: str) -> str:
    if kind == "counter":
        return "counter"
    elif kind == "histogram":
        return "histogram"
    elif kind == "gauge":
        return "gauge"
    return "untyped"
