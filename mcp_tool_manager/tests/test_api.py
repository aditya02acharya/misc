"""Tests for REST API endpoints: sync, health, metrics."""
import json
import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from mcp_tool_manager.api.sync import router as sync_router
from mcp_tool_manager.api.health import router as health_router
from mcp_tool_manager.api.metrics import router as metrics_router
from mcp_tool_manager.models import SyncJobStatus, SyncResult


def make_test_app():
    app = FastAPI()
    app.include_router(sync_router, prefix="/v1")
    app.include_router(health_router, prefix="/v1")
    app.include_router(metrics_router, prefix="/v1")
    return app


# ---- Metrics tests (no external deps) ----

def test_metrics_endpoint():
    app = make_test_app()
    with TestClient(app) as client:
        response = client.get("/v1/metrics")
    assert response.status_code == 200
    assert "mcp_tool_calls_count" in response.text
    assert "mcp_search_count" in response.text
    assert "# HELP" in response.text
    assert "# TYPE" in response.text


def test_metrics_prometheus_format():
    app = make_test_app()
    with TestClient(app) as client:
        response = client.get("/v1/metrics")
    lines = response.text.strip().split("\n")
    # Every metric should have HELP, TYPE, and value lines
    assert len(lines) >= 3


# ---- Sync API tests ----

@pytest.mark.asyncio
async def test_sync_trigger():
    app = make_test_app()

    mock_job_id = "sync_abc12345"
    with patch(
        "mcp_tool_manager.api.sync.get_redis",
        return_value=AsyncMock(),
    ), patch(
        "mcp_tool_manager.api.sync.get_httpx",
        return_value=MagicMock(),
    ), patch(
        "mcp_tool_manager.api.sync.start_sync",
        new_callable=AsyncMock,
        return_value=(mock_job_id, "running"),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/sync", json={"force": False})

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == mock_job_id
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_sync_get_status_found():
    app = make_test_app()
    job = SyncJobStatus(
        job_id="sync_abc12345",
        status="completed",
        stats=SyncResult(fetched=10, created=5, unchanged=5),
    )

    with patch("mcp_tool_manager.api.sync.get_redis", return_value=AsyncMock()), patch(
        "mcp_tool_manager.api.sync.get_sync_job_status",
        new_callable=AsyncMock,
        return_value=job,
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/sync/sync_abc12345")

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "sync_abc12345"
    assert data["status"] == "completed"


@pytest.mark.asyncio
async def test_sync_get_status_not_found():
    app = make_test_app()

    with patch("mcp_tool_manager.api.sync.get_redis", return_value=AsyncMock()), patch(
        "mcp_tool_manager.api.sync.get_sync_job_status",
        new_callable=AsyncMock,
        return_value=None,
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/sync/nonexistent")

    assert response.status_code == 404


# ---- Health tests ----

@pytest.mark.asyncio
async def test_health_all_ok():
    app = make_test_app()

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.scan = AsyncMock(return_value=(0, [b"tool:github:search"]))
    mock_redis.get = AsyncMock(return_value=None)

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))

    with patch("mcp_tool_manager.api.health.get_redis", return_value=mock_redis), patch(
        "mcp_tool_manager.api.health.get_httpx", return_value=mock_http
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "redis" in data
    assert "tool_count" in data


@pytest.mark.asyncio
async def test_health_redis_down():
    app = make_test_app()

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=Exception("Connection refused"))

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))

    with patch("mcp_tool_manager.api.health.get_redis", return_value=mock_redis), patch(
        "mcp_tool_manager.api.health.get_httpx", return_value=mock_http
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/health")

    data = response.json()
    assert data["status"] == "degraded"
    assert "error" in data["redis"]
