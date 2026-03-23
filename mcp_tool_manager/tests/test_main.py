"""Tests for main.py: app creation and lifespan."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_create_app_returns_fastapi():
    from mcp_tool_manager.main import create_app
    app = create_app()
    assert isinstance(app, FastAPI)


def test_create_app_has_routes():
    from mcp_tool_manager.main import create_app
    app = create_app()
    paths = [route.path for route in app.routes]
    assert "/v1/sync" in paths
    assert "/v1/health" in paths
    assert "/v1/metrics" in paths


def test_create_app_title():
    from mcp_tool_manager.config import get_settings
    import mcp_tool_manager.config as cfg
    cfg._settings = None
    settings = get_settings()

    from mcp_tool_manager.main import create_app
    app = create_app()
    assert app.title == settings.app_name


@pytest.mark.asyncio
async def test_lifespan_startup_and_shutdown():
    """Test the full lifespan: startup initializes, shutdown cleans up."""
    mock_queue = asyncio.Queue(maxsize=10)
    mock_bedrock = MagicMock()

    mock_index = AsyncMock()
    mock_index.set_client = AsyncMock()
    mock_index.create = AsyncMock()

    async def fake_cache_writer(queue, settings):
        await asyncio.sleep(100)

    with patch("mcp_tool_manager.main.setup_telemetry"), \
         patch("mcp_tool_manager.main.init_dependencies", new_callable=AsyncMock), \
         patch("mcp_tool_manager.main.init_redis_indexes", new_callable=AsyncMock), \
         patch("mcp_tool_manager.dependencies.get_cache_queue", return_value=mock_queue), \
         patch("mcp_tool_manager.services.cache.cache_writer_task", fake_cache_writer), \
         patch("mcp_tool_manager.main.close_dependencies", new_callable=AsyncMock), \
         patch("mcp_tool_manager.services.executor.close_all_sessions", new_callable=AsyncMock):

        from mcp_tool_manager.main import lifespan, create_app
        app = create_app()

        async with lifespan(app):
            pass  # Startup and shutdown both run


@pytest.mark.asyncio
async def test_lifespan_close_sessions_exception_ignored():
    """close_all_sessions exception during shutdown is silently ignored."""
    mock_queue = asyncio.Queue()

    async def fake_cache_writer(queue, settings):
        await asyncio.sleep(100)

    async def failing_close():
        raise RuntimeError("session close failed")

    with patch("mcp_tool_manager.main.setup_telemetry"), \
         patch("mcp_tool_manager.main.init_dependencies", new_callable=AsyncMock), \
         patch("mcp_tool_manager.main.init_redis_indexes", new_callable=AsyncMock), \
         patch("mcp_tool_manager.dependencies.get_cache_queue", return_value=mock_queue), \
         patch("mcp_tool_manager.services.cache.cache_writer_task", fake_cache_writer), \
         patch("mcp_tool_manager.main.close_dependencies", new_callable=AsyncMock), \
         patch("mcp_tool_manager.services.executor.close_all_sessions", failing_close):

        from mcp_tool_manager.main import lifespan, create_app
        app = create_app()
        async with lifespan(app):
            pass  # Should not raise
