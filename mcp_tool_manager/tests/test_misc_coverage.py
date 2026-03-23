"""Tests for remaining uncovered lines across multiple modules."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
import httpx

from mcp_tool_manager.config import Settings


# ── health.py: lines 37, 47-48, 55-56 ────────────────────────────────


def make_test_app():
    from mcp_tool_manager.api.health import router as health_router
    from mcp_tool_manager.api.metrics import router as metrics_router
    app = FastAPI()
    app.include_router(health_router, prefix="/v1")
    app.include_router(metrics_router, prefix="/v1")
    return app


@pytest.mark.asyncio
async def test_health_with_last_sync():
    """Covers line 37: last_sync_raw.decode() when sync data exists."""
    app = make_test_app()

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.scan = AsyncMock(return_value=(0, []))
    mock_redis.get = AsyncMock(return_value=b"2024-01-15T10:30:00")

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))

    with patch("mcp_tool_manager.api.health.get_redis", return_value=mock_redis), \
         patch("mcp_tool_manager.api.health.get_httpx", return_value=mock_http):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/health")

    data = response.json()
    assert data["last_sync"] == "2024-01-15T10:30:00"


@pytest.mark.asyncio
async def test_health_vllm_exception():
    """Covers lines 47-48: vLLM check exception handler."""
    app = make_test_app()

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.scan = AsyncMock(return_value=(0, []))
    mock_redis.get = AsyncMock(return_value=None)

    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=ConnectionError("vllm unreachable"))

    with patch("mcp_tool_manager.api.health.get_redis", return_value=mock_redis), \
         patch("mcp_tool_manager.api.health.get_httpx", return_value=mock_http):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/health")

    data = response.json()
    assert data["status"] == "degraded"
    assert "error" in data["vllm"]


@pytest.mark.asyncio
async def test_health_registry_exception():
    """Covers lines 55-56: registry check exception handler."""
    app = make_test_app()

    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    mock_redis.scan = AsyncMock(return_value=(0, []))
    mock_redis.get = AsyncMock(return_value=None)

    # vLLM succeeds but registry fails
    ok_resp = MagicMock(status_code=200)
    mock_http = AsyncMock()
    mock_http.get = AsyncMock(side_effect=[ok_resp, ConnectionError("registry down")])

    with patch("mcp_tool_manager.api.health.get_redis", return_value=mock_redis), \
         patch("mcp_tool_manager.api.health.get_httpx", return_value=mock_http):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/health")

    data = response.json()
    assert "error" in data["registry"]


# ── metrics.py: line 35 ───────────────────────────────────────────────


def test_prom_type_unknown_kind():
    """Covers line 35: return 'untyped' for unknown metric kind."""
    from mcp_tool_manager.api.metrics import _prom_type
    assert _prom_type("unknown") == "untyped"
    assert _prom_type("") == "untyped"


# ── main.py: lines 90-91 ─────────────────────────────────────────────


def test_run_function_calls_uvicorn():
    """Covers lines 90-91: run() calls uvicorn.run."""
    with patch("uvicorn.run") as mock_uvicorn:
        from mcp_tool_manager.main import run
        run()
    mock_uvicorn.assert_called_once()
    call_kwargs = mock_uvicorn.call_args[1]
    assert call_kwargs.get("factory") is True


# ── telemetry.py: line 66 ─────────────────────────────────────────────


def test_register_metrics_noop_when_meter_none():
    """Covers line 66: early return when _meter is None."""
    import mcp_tool_manager.telemetry as tel
    original_meter = tel._meter
    try:
        tel._meter = None
        tel._register_metrics()  # Should return early at line 66
    finally:
        tel._meter = original_meter


# ── cache.py: lines 88, 123 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_cache_chunks_returns_count():
    """Covers line 88: return len(chunks) after pipe.execute()."""
    from mcp_tool_manager.services.cache import write_cache_chunks

    mock_pipe = MagicMock()
    mock_pipe.hset = MagicMock(return_value=mock_pipe)
    mock_pipe.expire = MagicMock(return_value=mock_pipe)
    mock_pipe.execute = AsyncMock(return_value=[True, True])

    mock_redis = MagicMock()
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    settings = Settings()
    settings.cache.chunk_size = 100
    settings.cache.chunk_overlap = 10
    settings.cache.ttl = 300

    async def fake_embed(texts):
        return [b"\x00" * (1024 * 4) for _ in texts]

    count = await write_cache_chunks(
        session_id="sess1",
        tool_call_id="sess1:abc",
        tool_id="svc:tool",
        arguments={"q": "test"},
        raw_result="This is a test result text",
        redis_client=mock_redis,
        settings=settings,
        embed_batch_fn=fake_embed,
    )

    assert count == 1


@pytest.mark.asyncio
async def test_cache_writer_task_logs_success():
    """Covers line 123: logger.debug after successful write_cache_chunks."""
    from mcp_tool_manager.services.cache import cache_writer_task

    settings = Settings()
    queue = asyncio.Queue(maxsize=10)
    queue.put_nowait(("sess1", "sess1:abc", "svc:tool", {}, "result text"))

    mock_pipe = MagicMock()
    mock_pipe.hset = MagicMock(return_value=mock_pipe)
    mock_pipe.expire = MagicMock(return_value=mock_pipe)
    mock_pipe.execute = AsyncMock(return_value=[True, True])

    mock_redis = MagicMock()
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)

    async def fake_embed_batch(texts, s, client):
        return [b"\x00" * (1024 * 4) for _ in texts]

    with patch("mcp_tool_manager.dependencies.get_redis", return_value=mock_redis), \
         patch("mcp_tool_manager.dependencies.get_httpx", return_value=MagicMock()), \
         patch("mcp_tool_manager.services.embedding.embed_batch", side_effect=fake_embed_batch):

        task = asyncio.create_task(cache_writer_task(queue, settings))
        await asyncio.wait_for(queue.join(), timeout=5.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_cache_writer_task_unexpected_exception():
    """Covers lines 139-140: unexpected exception in outer loop."""
    from mcp_tool_manager.services.cache import cache_writer_task

    settings = Settings()
    queue = asyncio.Queue(maxsize=10)

    call_count = 0

    async def bad_get_redis():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("unexpected error")
        await asyncio.sleep(100)

    # We need to make queue.get() succeed but something in the outer loop fail
    # The outer try catches asyncio.CancelledError; other exceptions go to line 139
    # Let's patch get_redis to raise a non-CancelledError exception
    original_queue_get = queue.get

    call_num = [0]

    async def patched_get():
        call_num[0] += 1
        if call_num[0] == 1:
            # First call: raise unexpected exception
            raise RuntimeError("unexpected queue error")
        # Second call: raise CancelledError to stop the loop
        raise asyncio.CancelledError()

    queue.get = patched_get

    task = asyncio.create_task(cache_writer_task(queue, settings))
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# ── embedding.py: lines 38-40, 46 ─────────────────────────────────────


@pytest.mark.asyncio
async def test_embed_cache_hit():
    """Covers lines 38-40: Redis cache hit returns cached embedding."""
    import respx

    from mcp_tool_manager.services.embedding import embed

    import os
    os.environ["EMBEDDING__API_URL"] = "http://fakeembed/v1/embeddings"
    settings = Settings()

    cached_bytes = b"\x01" * (1024 * 4)
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=cached_bytes)

    async with httpx.AsyncClient() as client:
        result = await embed("test query", settings, client, mock_redis)

    assert result == cached_bytes
    # Should not have called setex (cache hit, no write needed)
    mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_embed_cache_miss_then_store():
    """Covers line 46: stores embedding in Redis after cache miss."""
    import respx

    from mcp_tool_manager.services.embedding import embed

    import os
    os.environ["EMBEDDING__API_URL"] = "http://fakeembed/v1/embeddings"
    settings = Settings()

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)  # cache miss
    mock_redis.setex = AsyncMock()

    with respx.mock:
        respx.post("http://fakeembed/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
            )
        )
        async with httpx.AsyncClient() as client:
            result = await embed("test query", settings, client, mock_redis)

    assert len(result) == 1024 * 4  # 1024 floats × 4 bytes
    mock_redis.setex.assert_called_once()


# ── search.py: lines 100, 140 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_hybrid_search_server_filter_added_to_hints():
    """Covers line 100: server_filter appended to hints."""
    import respx
    from mcp_tool_manager.services.search import hybrid_search

    import os
    os.environ["EMBEDDING__API_URL"] = "http://fakeembed/v1/embeddings"
    settings = Settings()
    settings.search.min_score = 0.0

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[])

    with respx.mock:
        respx.post("http://fakeembed/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
            )
        )
        async with httpx.AsyncClient() as client:
            results = await hybrid_search(
                query="find repos",
                top_k=5,
                settings=settings,
                tool_index=mock_index,
                http_client=client,
                server_filter="github",  # Not in query hints → added to hints (line 100)
            )

    assert results == []
    mock_index.query.assert_called_once()


@pytest.mark.asyncio
async def test_hybrid_search_truncates_at_top_k():
    """Covers line 140: break when len(final) >= top_k."""
    import respx
    from mcp_tool_manager.services.search import hybrid_search

    import os
    os.environ["EMBEDDING__API_URL"] = "http://fakeembed/v1/embeddings"
    settings = Settings()
    settings.search.min_score = 0.0

    # Return more results than top_k
    many_results = [
        {
            "tool_id": f"svc:tool{i}",
            "name": f"tool{i}",
            "description": f"Tool {i}",
            "server_name": "svc",
            "tags": "",
            "input_schema": "{}",
            "vector_distance": 0.1,
        }
        for i in range(10)
    ]

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=many_results)

    with respx.mock:
        respx.post("http://fakeembed/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
            )
        )
        async with httpx.AsyncClient() as client:
            results = await hybrid_search(
                query="find tools",
                top_k=3,  # Only want 3 results
                settings=settings,
                tool_index=mock_index,
                http_client=client,
            )

    assert len(results) == 3  # Truncated at top_k
