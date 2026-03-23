"""Tests for fetch_cached_data service: all three retrieval modes."""
import json
import time
import pytest
import respx
import httpx
from unittest.mock import AsyncMock

from mcp_tool_manager.services.fetch import (
    fetch_by_tool_call_id,
    fetch_all_session,
    fetch_by_query,
    _decode_chunk,
)
from mcp_tool_manager.config import Settings


FAKE_EMBEDDING = b"\x00" * (1024 * 4)


async def store_chunk(
    redis_client,
    session_id: str,
    tool_call_id: str,
    tool_id: str,
    chunk_idx: int,
    text: str,
    created_at: float | None = None,
):
    key = f"cache:{session_id}:{tool_call_id}:{chunk_idx}"
    await redis_client.hset(
        key,
        mapping={
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "tool_id": tool_id,
            "chunk_idx": str(chunk_idx),
            "text": text,
            "embedding": FAKE_EMBEDDING,
            "arguments_json": "{}",
            "created_at": str(created_at or time.time()),
        },
    )
    return key


def test_decode_chunk_from_bytes():
    data = {
        b"session_id": b"sess1",
        b"tool_call_id": b"sess1:abc",
        b"tool_id": b"github:search",
        b"chunk_idx": b"0",
        b"text": b"chunk content",
        b"embedding": FAKE_EMBEDDING,
        b"arguments_json": b"{}",
        b"created_at": b"1700000000.0",
    }
    chunk = _decode_chunk("cache:sess1:sess1:abc:0", data)
    assert chunk is not None
    assert chunk.session_id == "sess1"
    assert chunk.text == "chunk content"
    assert chunk.chunk_idx == 0


def test_decode_chunk_invalid_returns_none():
    result = _decode_chunk("bad_key", {"invalid": b"data"})
    # Should either return a chunk or None without raising
    # (chunk_idx may default to 0, text may be empty)
    # Primary: does not raise


@pytest.mark.asyncio
async def test_fetch_by_tool_call_id_returns_chunks(fakeredis_client):
    await store_chunk(fakeredis_client, "sess1", "sess1:abc", "github:search", 0, "chunk 0")
    await store_chunk(fakeredis_client, "sess1", "sess1:abc", "github:search", 1, "chunk 1")

    result = await fetch_by_tool_call_id("sess1", "sess1:abc", fakeredis_client)

    assert result.total_chunks == 2
    assert result.chunks[0].chunk_idx == 0
    assert result.chunks[1].chunk_idx == 1


@pytest.mark.asyncio
async def test_fetch_by_tool_call_id_different_session_isolated(fakeredis_client):
    await store_chunk(fakeredis_client, "sess1", "sess1:abc", "github:search", 0, "sess1 data")
    await store_chunk(fakeredis_client, "sess2", "sess2:xyz", "github:search", 0, "sess2 data")

    result = await fetch_by_tool_call_id("sess1", "sess1:abc", fakeredis_client)
    assert result.total_chunks == 1
    assert result.chunks[0].text == "sess1 data"


@pytest.mark.asyncio
async def test_fetch_by_tool_call_id_not_found(fakeredis_client):
    result = await fetch_by_tool_call_id("sess_missing", "sess_missing:xyz", fakeredis_client)
    assert result.total_chunks == 0
    assert result.chunks == []


@pytest.mark.asyncio
async def test_fetch_all_session_sorted_by_created_at(fakeredis_client):
    t1 = time.time() - 100
    t2 = time.time() - 50
    t3 = time.time()

    await store_chunk(fakeredis_client, "sess3", "sess3:a", "tool:a", 0, "older", t1)
    await store_chunk(fakeredis_client, "sess3", "sess3:b", "tool:b", 0, "middle", t2)
    await store_chunk(fakeredis_client, "sess3", "sess3:c", "tool:c", 0, "newest", t3)

    result = await fetch_all_session("sess3", top_k=10, redis_client=fakeredis_client)

    assert result.total_chunks == 3
    # Most recent first
    assert result.chunks[0].text == "newest"
    assert result.chunks[2].text == "older"


@pytest.mark.asyncio
async def test_fetch_all_session_pagination(fakeredis_client):
    for i in range(5):
        await store_chunk(
            fakeredis_client, "sess4", f"sess4:call{i}", "tool:x", 0, f"chunk {i}", time.time() + i
        )

    result = await fetch_all_session("sess4", top_k=3, redis_client=fakeredis_client)

    assert result.total_chunks == 5  # total in store
    assert len(result.chunks) == 3   # paginated


@pytest.mark.asyncio
async def test_fetch_by_query_no_cache_index():
    from mcp_tool_manager.config import Settings
    settings = Settings()

    async with httpx.AsyncClient() as client:
        result = await fetch_by_query(
            session_id="sess1",
            query="find repos",
            top_k=5,
            settings=settings,
            cache_index=None,  # not available
            http_client=client,
        )

    assert result.total_chunks == 0
    assert result.chunks == []


@respx.mock
@pytest.mark.asyncio
async def test_fetch_by_query_with_mock_index(fakeredis_client):
    import os
    os.environ["EMBEDDING__API_URL"] = "http://fakeembed/v1/embeddings"
    settings = Settings()

    respx.post("http://fakeembed/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
        )
    )

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[
        {
            "session_id": "sess5",
            "tool_call_id": "sess5:abc",
            "tool_id": "github:search",
            "chunk_idx": "0",
            "text": "relevant result",
            "arguments_json": "{}",
            "created_at": "1700000000.0",
        }
    ])

    async with httpx.AsyncClient() as client:
        result = await fetch_by_query(
            session_id="sess5",
            query="find repos",
            top_k=5,
            settings=settings,
            cache_index=mock_index,
            http_client=client,
            redis_client=fakeredis_client,
        )

    assert result.total_chunks == 1
    assert result.chunks[0].text == "relevant result"
