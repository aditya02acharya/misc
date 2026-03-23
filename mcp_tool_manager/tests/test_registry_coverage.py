"""Coverage tests for registry.py using AsyncMock (fixes async coverage tracking)."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from mcp_tool_manager.services.registry import (
    fetch_servers,
    fetch_tools_for_server,
    run_sync_job,
    start_sync,
    get_sync_job_status,
    _compose_embedding_text,
)
from mcp_tool_manager.models import ToolDoc, SyncJobStatus, SyncResult
from mcp_tool_manager.config import Settings

FAKE_VECTOR = b"\x00" * (1024 * 4)


def make_settings(api_key=None):
    s = Settings()
    s.registry.api_url = "http://fakeregistry"
    if api_key:
        s.registry.api_key = api_key
    return s


def _make_mock_http(servers_json, tools_json=None):
    """Create an AsyncMock http client that returns pre-configured responses."""
    mock_http = AsyncMock()
    responses = []

    servers_resp = MagicMock()
    servers_resp.raise_for_status = MagicMock()
    servers_resp.json = MagicMock(return_value=servers_json)
    responses.append(servers_resp)

    if tools_json is not None:
        tools_resp = MagicMock()
        tools_resp.raise_for_status = MagicMock()
        tools_resp.json = MagicMock(return_value=tools_json)
        responses.append(tools_resp)

    mock_http.get = AsyncMock(side_effect=responses)
    return mock_http


def _make_mock_redis(existing_hashes=None, scan_keys=None):
    """Create an AsyncMock redis client."""
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)

    # Pipeline mock
    mock_pipe = MagicMock()
    mock_pipe.hget = MagicMock(return_value=mock_pipe)
    mock_pipe.hset = MagicMock(return_value=mock_pipe)
    mock_pipe.expire = MagicMock(return_value=mock_pipe)
    hash_returns = existing_hashes or [None]
    mock_pipe.execute = AsyncMock(side_effect=[hash_returns, [True] * len(hash_returns)])
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    # SCAN mock
    keys = scan_keys or []
    mock_redis.scan = AsyncMock(return_value=(0, keys))

    return mock_redis


# ── fetch_servers ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_servers_with_api_key():
    """Covers line 52: Authorization header added when api_key set."""
    settings = make_settings(api_key="secret-key")

    servers_resp = MagicMock()
    servers_resp.raise_for_status = MagicMock()
    servers_resp.json = MagicMock(return_value={"servers": [{"server_name": "svc"}]})

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=servers_resp)

    result = await fetch_servers(settings, mock_http)
    assert result == [{"server_name": "svc"}]

    # Verify Authorization header was passed
    call_kwargs = mock_http.get.call_args
    headers = call_kwargs[1].get("headers", {}) if call_kwargs[1] else call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {}
    assert "Authorization" in headers or any("Authorization" in str(c) for c in mock_http.get.call_args_list)


@pytest.mark.asyncio
async def test_fetch_servers_list_response():
    """Covers line 61: response is a list (not dict)."""
    settings = make_settings()

    servers_resp = MagicMock()
    servers_resp.raise_for_status = MagicMock()
    servers_resp.json = MagicMock(return_value=[{"server_name": "svc1"}, {"server_name": "svc2"}])

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=servers_resp)

    result = await fetch_servers(settings, mock_http)
    assert len(result) == 2


# ── fetch_tools_for_server ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_tools_for_server_with_api_key():
    """Covers line 74: Authorization header added."""
    settings = make_settings(api_key="secret")
    server = {"server_name": "github", "description": "GitHub", "url": "http://gh"}

    tools_resp = MagicMock()
    tools_resp.raise_for_status = MagicMock()
    tools_resp.json = MagicMock(return_value={"tools": [
        {"name": "search", "description": "Search", "inputSchema": {}}
    ]})

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=tools_resp)

    tools = await fetch_tools_for_server(server, settings, mock_http)
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_fetch_tools_for_server_exception():
    """Covers lines 85-87: exception returns empty list."""
    settings = make_settings()
    server = {"server_name": "broken", "description": "", "url": "http://broken"}

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=ConnectionError("refused"))

    tools = await fetch_tools_for_server(server, settings, mock_http)
    assert tools == []


# ── run_sync_job ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_sync_job_new_tools_mock():
    """Covers lines 131-229: full sync with new tools to embed."""
    settings = make_settings()

    mock_redis = _make_mock_redis(existing_hashes=[None])  # no existing hash
    mock_http = _make_mock_http(
        servers_json={"servers": [{"server_name": "github", "description": "GH", "url": "http://gh"}]},
        tools_json={"tools": [{"name": "search", "description": "Search repos", "inputSchema": {"type": "object"}}]},
    )

    async def fake_embed(texts):
        return [FAKE_VECTOR for _ in texts]

    await run_sync_job(
        job_id="test_new_tools",
        force=False,
        settings=settings,
        http_client=mock_http,
        redis_client=mock_redis,
        embed_batch_fn=fake_embed,
    )

    # Status should have been updated at least twice (start + complete)
    assert mock_redis.setex.call_count >= 2


@pytest.mark.asyncio
async def test_run_sync_job_unchanged_tools_mock():
    """Covers lines 157-162: unchanged tools are skipped."""
    from mcp_tool_manager.services.registry import _content_hash
    settings = make_settings()

    chash = _content_hash("github", "search", "Search repos", {"type": "object"})
    mock_redis = _make_mock_redis(existing_hashes=[chash.encode()])
    mock_http = _make_mock_http(
        servers_json={"servers": [{"server_name": "github", "description": "GH", "url": "http://gh"}]},
        tools_json={"tools": [{"name": "search", "description": "Search repos", "inputSchema": {"type": "object"}}]},
    )

    async def fake_embed(texts):
        return [FAKE_VECTOR for _ in texts]

    await run_sync_job(
        job_id="test_unchanged",
        force=False,
        settings=settings,
        http_client=mock_http,
        redis_client=mock_redis,
        embed_batch_fn=fake_embed,
    )

    assert mock_redis.setex.call_count >= 2


@pytest.mark.asyncio
async def test_run_sync_job_deletes_stale_mock():
    """Covers lines 200-212: stale tool keys are deleted."""
    settings = make_settings()

    stale_key = b"tool:old:stale"
    mock_redis = _make_mock_redis(existing_hashes=[None], scan_keys=[stale_key])
    mock_http = _make_mock_http(
        servers_json={"servers": [{"server_name": "github", "description": "GH", "url": "http://gh"}]},
        tools_json={"tools": [{"name": "search", "description": "Search", "inputSchema": {}}]},
    )

    async def fake_embed(texts):
        return [FAKE_VECTOR for _ in texts]

    await run_sync_job(
        job_id="test_delete_stale",
        force=False,
        settings=settings,
        http_client=mock_http,
        redis_client=mock_redis,
        embed_batch_fn=fake_embed,
    )

    mock_redis.delete.assert_called()


@pytest.mark.asyncio
async def test_run_sync_job_failure_mock():
    """Covers lines 221-226: exception transitions job to failed."""
    settings = make_settings()

    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=Exception("registry down"))

    async def fake_embed(texts):
        return []

    await run_sync_job(
        job_id="test_failure",
        force=False,
        settings=settings,
        http_client=mock_http,
        redis_client=mock_redis,
        embed_batch_fn=fake_embed,
    )

    # Final setex should include failed status
    last_call = mock_redis.setex.call_args_list[-1]
    stored_json = last_call[0][2] if last_call[0] else last_call[1].get("value", "{}")
    assert "failed" in stored_json


@pytest.mark.asyncio
async def test_run_sync_job_empty_server_list_mock():
    """Covers lines 162-171: no tools → embed step skipped."""
    settings = make_settings()

    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    mock_redis.scan = AsyncMock(return_value=(0, []))
    mock_redis.pipeline = MagicMock(return_value=MagicMock(execute=AsyncMock(return_value=[])))

    mock_http = _make_mock_http(servers_json={"servers": []})

    async def fake_embed(texts):
        return []

    await run_sync_job(
        job_id="test_empty",
        force=False,
        settings=settings,
        http_client=mock_http,
        redis_client=mock_redis,
        embed_batch_fn=fake_embed,
    )

    assert mock_redis.setex.call_count >= 2


@pytest.mark.asyncio
async def test_run_sync_job_updated_tools_mock():
    """Covers line 181: stats.updated incremented for changed hash."""
    settings = make_settings()

    # Existing hash is different from current
    mock_redis = _make_mock_redis(existing_hashes=[b"old_hash_value"])
    mock_http = _make_mock_http(
        servers_json={"servers": [{"server_name": "github", "description": "GH", "url": "http://gh"}]},
        tools_json={"tools": [{"name": "search", "description": "Search repos", "inputSchema": {}}]},
    )

    async def fake_embed(texts):
        return [FAKE_VECTOR for _ in texts]

    await run_sync_job(
        job_id="test_updated",
        force=False,
        settings=settings,
        http_client=mock_http,
        redis_client=mock_redis,
        embed_batch_fn=fake_embed,
    )

    assert mock_redis.setex.call_count >= 2


# ── start_sync ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_sync_new_job_mock():
    """Covers lines 250-262: acquires lock and creates task."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)   # no existing lock
    mock_redis.set = AsyncMock(return_value=True)   # lock acquired

    with patch("asyncio.create_task"):
        job_id, status = await start_sync(
            force=False,
            settings=make_settings(),
            http_client=AsyncMock(),
            redis_client=mock_redis,
            embed_batch_fn=AsyncMock(return_value=[]),
        )

    assert status == "running"
    assert job_id.startswith("sync_")


@pytest.mark.asyncio
async def test_start_sync_existing_lock_mock():
    """Covers lines 245-247: returns existing lock job_id."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"sync_existing_job")

    job_id, status = await start_sync(
        force=False,
        settings=make_settings(),
        http_client=AsyncMock(),
        redis_client=mock_redis,
        embed_batch_fn=AsyncMock(return_value=[]),
    )

    assert job_id == "sync_existing_job"
    assert status == "running"


@pytest.mark.asyncio
async def test_start_sync_race_condition_mock():
    """Covers lines 252-256: lock acquired by another process."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(side_effect=[None, b"sync_other_job"])
    mock_redis.set = AsyncMock(return_value=False)  # lock not acquired (race)

    job_id, status = await start_sync(
        force=False,
        settings=make_settings(),
        http_client=AsyncMock(),
        redis_client=mock_redis,
        embed_batch_fn=AsyncMock(return_value=[]),
    )

    assert job_id == "sync_other_job"
    assert status == "running"


@pytest.mark.asyncio
async def test_start_sync_force_overrides_lock_mock():
    """Covers force=True path that bypasses existing lock check."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=b"sync_old_job")
    mock_redis.set = AsyncMock(return_value=True)

    with patch("asyncio.create_task"):
        job_id, status = await start_sync(
            force=True,
            settings=make_settings(),
            http_client=AsyncMock(),
            redis_client=mock_redis,
            embed_batch_fn=AsyncMock(return_value=[]),
        )

    assert status == "running"


# ── get_sync_job_status ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_sync_job_status_found_mock():
    """Covers lines 268-270: returns parsed SyncJobStatus."""
    job = SyncJobStatus(job_id="sync_test01", status="completed")
    job_json = job.model_dump_json()

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=job_json.encode())

    from mcp_tool_manager.services.registry import get_sync_job_status
    result = await get_sync_job_status("sync_test01", mock_redis)

    assert result is not None
    assert result.job_id == "sync_test01"
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_get_sync_job_status_not_found_mock():
    """Covers line 269: returns None when no data."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    from mcp_tool_manager.services.registry import get_sync_job_status
    result = await get_sync_job_status("nonexistent", mock_redis)
    assert result is None


# ── _compose_embedding_text edge cases ─────────────────────────────


def test_compose_embedding_text_more_than_5_params():
    """Covers line 33: break when params > 5."""
    doc = ToolDoc(
        tool_id="svc:tool",
        name="tool",
        description="Test",
        server_name="svc",
        server_description="",
        tags=[],
        input_schema={
            "properties": {
                "p1": {"description": "Param 1"},
                "p2": {"description": "Param 2"},
                "p3": {"description": "Param 3"},
                "p4": {"description": "Param 4"},
                "p5": {"description": "Param 5"},
                "p6": {"description": "Param 6"},  # Should be cut off
            }
        },
    )
    text = _compose_embedding_text(doc)
    assert "p5" in text
    assert "p6" not in text
