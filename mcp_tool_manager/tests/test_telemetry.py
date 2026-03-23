"""Tests for telemetry: setup, metric recording, tracer."""
import pytest
from unittest.mock import MagicMock, patch
from mcp_tool_manager.config import Settings


def _reset_telemetry():
    import mcp_tool_manager.telemetry as t
    t._meter = None
    t._tracer = None
    t._metric_instruments = {}


def test_setup_telemetry_disabled():
    _reset_telemetry()
    settings = Settings()
    settings.otel.enabled = False

    from mcp_tool_manager.telemetry import setup_telemetry, get_metric_instruments
    setup_telemetry(settings)

    instruments = get_metric_instruments()
    assert len(instruments) > 0


def test_setup_telemetry_enabled_phoenix_fails():
    _reset_telemetry()
    settings = Settings()
    settings.otel.enabled = True
    settings.otel.api_key = "test-key"

    with patch.dict("sys.modules", {"phoenix.otel": MagicMock(register=MagicMock(side_effect=Exception("no phoenix")))}):
        from mcp_tool_manager.telemetry import setup_telemetry, get_metric_instruments
        setup_telemetry(settings)  # Should not raise

    instruments = get_metric_instruments()
    assert len(instruments) > 0


def test_setup_telemetry_enabled_phoenix_ok():
    _reset_telemetry()
    settings = Settings()
    settings.otel.enabled = True

    mock_register = MagicMock()
    mock_phoenix = MagicMock()
    mock_phoenix.register = mock_register

    with patch.dict("sys.modules", {"phoenix": mock_phoenix, "phoenix.otel": mock_phoenix}):
        from mcp_tool_manager.telemetry import setup_telemetry
        setup_telemetry(settings)

    # Should complete without error
    from mcp_tool_manager.telemetry import get_metric_instruments
    assert len(get_metric_instruments()) > 0


def test_register_metrics_all_types():
    _reset_telemetry()
    settings = Settings()
    settings.otel.enabled = False

    from mcp_tool_manager.telemetry import setup_telemetry, get_metric_instruments, METRICS_REGISTRY
    setup_telemetry(settings)

    instruments = get_metric_instruments()
    # All metrics should be registered
    for name in METRICS_REGISTRY:
        assert name in instruments


def test_get_tracer_when_none():
    _reset_telemetry()
    from mcp_tool_manager import telemetry
    telemetry._tracer = None
    tracer = telemetry.get_tracer()
    assert tracer is not None


def test_get_tracer_when_set():
    from mcp_tool_manager import telemetry
    mock_tracer = MagicMock()
    telemetry._tracer = mock_tracer
    result = telemetry.get_tracer()
    assert result is mock_tracer


def test_record_counter_with_instrument():
    _reset_telemetry()
    settings = Settings()
    settings.otel.enabled = False
    from mcp_tool_manager.telemetry import setup_telemetry, record_counter, _metric_instruments
    setup_telemetry(settings)

    # Should not raise
    record_counter("mcp.search.count")
    record_counter("mcp.search.count", value=5)
    record_counter("mcp.search.count", attributes={"key": "val"})


def test_record_counter_missing_name():
    from mcp_tool_manager.telemetry import record_counter
    # Missing metric name should not raise
    record_counter("mcp.nonexistent.metric")


def test_record_histogram_with_instrument():
    _reset_telemetry()
    settings = Settings()
    settings.otel.enabled = False
    from mcp_tool_manager.telemetry import setup_telemetry, record_histogram
    setup_telemetry(settings)

    record_histogram("mcp.search.latency_ms", 42.5)
    record_histogram("mcp.search.latency_ms", 10.0, attributes={"op": "test"})


def test_record_histogram_missing_name():
    from mcp_tool_manager.telemetry import record_histogram
    record_histogram("mcp.nonexistent", 1.0)


def test_record_gauge_with_instrument():
    _reset_telemetry()
    settings = Settings()
    settings.otel.enabled = False
    from mcp_tool_manager.telemetry import setup_telemetry, record_gauge
    setup_telemetry(settings)

    record_gauge("mcp.tools.total", 150.0)
    record_gauge("mcp.cache.queue_depth", 5.0, attributes={"env": "test"})


def test_record_gauge_missing_name():
    from mcp_tool_manager.telemetry import record_gauge
    record_gauge("mcp.nonexistent.gauge", 1.0)
