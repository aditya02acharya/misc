"""GET /v1/metrics — Prometheus-compatible metrics endpoint."""
import logging
from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from mcp_tool_manager.telemetry import METRICS_REGISTRY

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Return Prometheus-compatible text format metrics from OTEL MeterProvider."""
    lines = []

    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry import metrics as otel_metrics

        provider = otel_metrics.get_meter_provider()
        if isinstance(provider, MeterProvider):
            # Collect metrics from all readers
            for reader in provider._all_metric_readers:
                metrics_data = reader.get_metrics_data()
                if metrics_data is None:
                    continue
                for resource_metrics in metrics_data.resource_metrics:
                    for scope_metrics in resource_metrics.scope_metrics:
                        for metric in scope_metrics.metrics:
                            prom_name = metric.name.replace(".", "_").replace("-", "_")
                            registry_entry = METRICS_REGISTRY.get(metric.name)
                            desc = registry_entry[2] if registry_entry else metric.description
                            kind = registry_entry[0] if registry_entry else "untyped"

                            lines.append(f"# HELP {prom_name} {desc}")
                            lines.append(f"# TYPE {prom_name} {_prom_type(kind)}")

                            for data_point in metric.data.data_points:
                                labels = _format_labels(data_point.attributes or {})
                                if hasattr(data_point, "value"):
                                    lines.append(f"{prom_name}{labels} {data_point.value}")
                                elif hasattr(data_point, "sum"):
                                    lines.append(f"{prom_name}_total{labels} {data_point.sum}")
                                    lines.append(f"{prom_name}_count{labels} {data_point.count}")

            if lines:
                return "\n".join(lines) + "\n"
    except Exception as exc:
        logger.debug("Could not read from MeterProvider: %s", exc)

    # Fallback: emit registry with zeros
    for name, (kind, unit, desc) in METRICS_REGISTRY.items():
        prom_name = name.replace(".", "_").replace("-", "_")
        lines.append(f"# HELP {prom_name} {desc}")
        lines.append(f"# TYPE {prom_name} {_prom_type(kind)}")
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


def _format_labels(attributes: dict) -> str:
    if not attributes:
        return ""
    pairs = [f'{k}="{v}"' for k, v in sorted(attributes.items())]
    return "{" + ",".join(pairs) + "}"
