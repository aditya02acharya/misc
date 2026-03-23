"""Coverage tests for fetch.py using AsyncMock (fixes async coverage tracking)."""
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_tool_manager.services.fetch import (
    _decode_chunk,
    fetch_by_tool_call_id,
    fetch_all_session,
    fetch_by_query,
)
from mcp_tool_manager.config import Settings

FAKE_EMBEDDING = b"\x00" * (1024 * 4)


def _chunk_data(session_id, tool_call_id, tool_id, chunk_idx, text, created_at=None):
    return {
        b"session_id": session_id.encode(),
        b"tool_call_id": tool_call_id.encode(),
        b"tool_id": tool_id.encode(),
        b"chunk_idx": str(chunk_idx).encode(),
        b"text": text.encode(),
        b"embedding": FAKE_EMBEDDING,
        b"arguments_json": b"{}",
        b"created_at": str(created_at or time.time()).encode(),
    }


# ── _decode_chunk exception path ─────────────────────────────────────


def test_decode_chunk_exception_returns_none():
    """Covers lines 28-30: exception in CacheChunk construction returns None."""
    # Pass data that will cause an exception during field parsing
    # chunk_idx can't be converted to int if it's a non-numeric non-empty value
    bad_data = {
        b"session_id": b"sess1",
        b"tool_call_id": b"sess1:abc",
        b"tool_id": b"svc:tool",
        b"chunk_idx": b"not_a_number",
        b"text": b"some text",
        b"embedding": FAKE_EMBEDDING,
        b"arguments_json": b"{}",
        b"created_at": b"also_not_a_number",  # float("") will raise
    }
    result = _decode_chunk("cache:sess1:sess1:abc:0", bad_data)
    assert result is None


# ── fetch_by_tool_call_id with AsyncMock ─────────────────────────────


@pytest.mark.asyncio
async def test_fetch_by_tool_call_id_with_mock():
    """Covers lines 44-56: scan loop and return."""
    chunk_data = _chunk_data("sess1", "sess1:abc", "github:search", 0, "result text")

    mock_redis = AsyncMock()
    key = b"cache:sess1:sess1:abc:0"
    mock_redis.scan = AsyncMock(return_value=(0, [key]))
    mock_redis.hgetall = AsyncMock(return_value=chunk_data)

    result = await fetch_by_tool_call_id("sess1", "sess1:abc", mock_redis)

    assert result.total_chunks == 1
    assert result.chunks[0].text == "result text"


@pytest.mark.asyncio
async def test_fetch_by_tool_call_id_multiple_chunks_mock():
    """Covers lines 44-56: multiple chunks sorted by chunk_idx."""
    key0 = b"cache:sess1:sess1:abc:0"
    key1 = b"cache:sess1:sess1:abc:1"

    data0 = _chunk_data("sess1", "sess1:abc", "svc:tool", 0, "first")
    data1 = _chunk_data("sess1", "sess1:abc", "svc:tool", 1, "second")

    mock_redis = AsyncMock()
    mock_redis.scan = AsyncMock(return_value=(0, [key1, key0]))  # reversed order
    mock_redis.hgetall = AsyncMock(side_effect=[data1, data0])

    result = await fetch_by_tool_call_id("sess1", "sess1:abc", mock_redis)

    assert result.total_chunks == 2
    assert result.chunks[0].chunk_idx == 0
    assert result.chunks[1].chunk_idx == 1


@pytest.mark.asyncio
async def test_fetch_by_tool_call_id_empty_mock():
    """Covers lines 51-52, 56: loop terminates immediately."""
    mock_redis = AsyncMock()
    mock_redis.scan = AsyncMock(return_value=(0, []))

    result = await fetch_by_tool_call_id("nosess", "nosess:call", mock_redis)

    assert result.total_chunks == 0
    assert result.chunks == []


@pytest.mark.asyncio
async def test_fetch_by_tool_call_id_decode_fails_mock():
    """Covers lines 45-50: bad chunk data is skipped (returns None)."""
    key = b"cache:sess1:sess1:abc:0"
    bad_data = {b"chunk_idx": b"bad", b"created_at": b"bad"}

    mock_redis = AsyncMock()
    mock_redis.scan = AsyncMock(return_value=(0, [key]))
    mock_redis.hgetall = AsyncMock(return_value=bad_data)

    result = await fetch_by_tool_call_id("sess1", "sess1:abc", mock_redis)
    # Bad chunk skipped → 0 total
    assert result.total_chunks == 0


@pytest.mark.asyncio
async def test_fetch_by_tool_call_id_string_key_mock():
    """Covers line 46: string key (not bytes) path."""
    key = "cache:sess1:sess1:abc:0"  # string, not bytes
    chunk_data = _chunk_data("sess1", "sess1:abc", "svc:tool", 0, "text")

    mock_redis = AsyncMock()
    mock_redis.scan = AsyncMock(return_value=(0, [key]))
    mock_redis.hgetall = AsyncMock(return_value=chunk_data)

    result = await fetch_by_tool_call_id("sess1", "sess1:abc", mock_redis)
    assert result.total_chunks == 1


# ── fetch_all_session with AsyncMock ─────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_all_session_with_mock():
    """Covers lines 133-148: full scan and sort."""
    t1 = time.time() - 100
    t2 = time.time() - 50

    key0 = b"cache:sess2:sess2:a:0"
    key1 = b"cache:sess2:sess2:b:0"
    data0 = _chunk_data("sess2", "sess2:a", "svc:tool", 0, "older", t1)
    data1 = _chunk_data("sess2", "sess2:b", "svc:tool", 0, "newer", t2)

    mock_redis = AsyncMock()
    mock_redis.scan = AsyncMock(return_value=(0, [key0, key1]))
    mock_redis.hgetall = AsyncMock(side_effect=[data0, data1])

    result = await fetch_all_session("sess2", top_k=10, redis_client=mock_redis)

    assert result.total_chunks == 2
    # Most recent first
    assert result.chunks[0].text == "newer"
    assert result.chunks[1].text == "older"


@pytest.mark.asyncio
async def test_fetch_all_session_pagination_mock():
    """Covers lines 147-148: top_k pagination."""
    keys = [f"cache:sess3:sess3:call{i}:0".encode() for i in range(5)]
    now = time.time()
    side_effects = [_chunk_data("sess3", f"sess3:call{i}", "svc:t", 0, f"text{i}", now + i) for i in range(5)]

    mock_redis = AsyncMock()
    mock_redis.scan = AsyncMock(return_value=(0, keys))
    mock_redis.hgetall = AsyncMock(side_effect=side_effects)

    result = await fetch_all_session("sess3", top_k=3, redis_client=mock_redis)

    assert result.total_chunks == 5
    assert len(result.chunks) == 3


@pytest.mark.asyncio
async def test_fetch_all_session_no_top_k_mock():
    """Covers line 147: top_k=0 means return all."""
    keys = [f"cache:sess4:sess4:call{i}:0".encode() for i in range(3)]
    now = time.time()
    side_effects = [_chunk_data("sess4", f"sess4:call{i}", "svc:t", 0, f"text{i}", now + i) for i in range(3)]

    mock_redis = AsyncMock()
    mock_redis.scan = AsyncMock(return_value=(0, keys))
    mock_redis.hgetall = AsyncMock(side_effect=side_effects)

    result = await fetch_all_session("sess4", top_k=0, redis_client=mock_redis)

    assert result.total_chunks == 3
    assert len(result.chunks) == 3


@pytest.mark.asyncio
async def test_fetch_all_session_skip_bad_chunks_mock():
    """Covers lines 137-139: None chunks are skipped."""
    good_key = b"cache:sess5:sess5:a:0"
    bad_key = b"cache:sess5:sess5:b:0"

    good_data = _chunk_data("sess5", "sess5:a", "svc:t", 0, "good text")
    bad_data = {b"chunk_idx": b"bad", b"created_at": b"not_a_float"}

    mock_redis = AsyncMock()
    mock_redis.scan = AsyncMock(return_value=(0, [good_key, bad_key]))
    mock_redis.hgetall = AsyncMock(side_effect=[good_data, bad_data])

    result = await fetch_all_session("sess5", top_k=10, redis_client=mock_redis)
    # Bad chunk skipped
    assert result.total_chunks == 1


# ── fetch_by_query exception path ─────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_by_query_exception_path():
    """Covers lines 98-100: exception returns empty FetchResult."""
    import os
    os.environ["EMBEDDING__API_URL"] = "http://fakeembed/v1/embeddings"
    settings = Settings()

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(side_effect=RuntimeError("query failed"))

    mock_http = AsyncMock()
    embed_vector = b"\x00" * (1024 * 4)

    with patch("mcp_tool_manager.services.embedding.embed", new_callable=AsyncMock, return_value=embed_vector):
        result = await fetch_by_query(
            session_id="sess6",
            query="test query",
            top_k=5,
            settings=settings,
            cache_index=mock_index,
            http_client=mock_http,
        )

    assert result.total_chunks == 0
    assert result.chunks == []
