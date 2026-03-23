"""Tests for dependencies: getters, init, indexes, close."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp_tool_manager.config import Settings


def _reset_deps():
    import mcp_tool_manager.dependencies as d
    d._redis_client = None
    d._httpx_client = None
    d._bedrock_client = None
    d._cache_queue = None
    d._tool_index = None
    d._cache_index = None


def test_get_redis_uninitialized():
    _reset_deps()
    from mcp_tool_manager.dependencies import get_redis
    with pytest.raises(RuntimeError, match="not initialized"):
        get_redis()


def test_get_httpx_uninitialized():
    _reset_deps()
    from mcp_tool_manager.dependencies import get_httpx
    with pytest.raises(RuntimeError, match="not initialized"):
        get_httpx()


def test_get_bedrock_uninitialized():
    _reset_deps()
    from mcp_tool_manager.dependencies import get_bedrock
    with pytest.raises(RuntimeError, match="not initialized"):
        get_bedrock()


def test_get_cache_queue_uninitialized():
    _reset_deps()
    from mcp_tool_manager.dependencies import get_cache_queue
    with pytest.raises(RuntimeError, match="not initialized"):
        get_cache_queue()


def test_get_tool_index_returns_none():
    _reset_deps()
    from mcp_tool_manager.dependencies import get_tool_index
    assert get_tool_index() is None


def test_get_cache_index_returns_none():
    _reset_deps()
    from mcp_tool_manager.dependencies import get_cache_index
    assert get_cache_index() is None


@pytest.mark.asyncio
async def test_init_dependencies(monkeypatch):
    _reset_deps()
    settings = Settings()
    settings.embedding.api_key = "test-key"
    settings.bedrock.endpoint_url = "http://localstack:4566"

    mock_bedrock = MagicMock()
    monkeypatch.setattr("boto3.client", lambda *a, **kw: mock_bedrock)

    from mcp_tool_manager.dependencies import init_dependencies, get_redis, get_httpx, get_bedrock, get_cache_queue
    await init_dependencies(settings)

    redis = get_redis()
    http = get_httpx()
    bedrock = get_bedrock()
    queue = get_cache_queue()

    assert redis is not None
    assert http is not None
    assert bedrock is mock_bedrock
    assert isinstance(queue, asyncio.Queue)

    # Cleanup
    await http.aclose()
    await redis.aclose()


@pytest.mark.asyncio
async def test_init_dependencies_no_api_key(monkeypatch):
    _reset_deps()
    settings = Settings()
    settings.embedding.api_key = ""

    monkeypatch.setattr("boto3.client", lambda *a, **kw: MagicMock())

    from mcp_tool_manager.dependencies import init_dependencies, get_httpx
    await init_dependencies(settings)
    http = get_httpx()
    assert http is not None
    await http.aclose()


@pytest.mark.asyncio
async def test_close_dependencies_cleans_up(monkeypatch):
    _reset_deps()
    settings = Settings()
    monkeypatch.setattr("boto3.client", lambda *a, **kw: MagicMock())

    from mcp_tool_manager.dependencies import init_dependencies, close_dependencies, get_redis
    await init_dependencies(settings)

    # Should not raise
    await close_dependencies()

    # Globals are cleaned up - getters should raise again or return closed clients
    # (Redis client ref remains but is closed)


@pytest.mark.asyncio
async def test_close_dependencies_when_not_initialized():
    _reset_deps()
    from mcp_tool_manager.dependencies import close_dependencies
    # Should not raise even if nothing initialized
    await close_dependencies()


@pytest.mark.asyncio
async def test_init_redis_indexes(fakeredis_client, monkeypatch):
    _reset_deps()

    # Inject fakeredis as the Redis client
    import mcp_tool_manager.dependencies as d
    d._redis_client = fakeredis_client

    settings = Settings()

    mock_index = AsyncMock()
    mock_index.set_client = AsyncMock()
    mock_index.create = AsyncMock()

    with patch("redisvl.index.SearchIndex") as MockSearchIndex:
        MockSearchIndex.from_dict.return_value = mock_index
        from mcp_tool_manager.dependencies import init_redis_indexes
        await init_redis_indexes(settings)

    assert MockSearchIndex.from_dict.call_count == 2
    assert mock_index.create.call_count == 2
    assert d._tool_index is mock_index
    assert d._cache_index is mock_index
