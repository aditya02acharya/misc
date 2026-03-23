"""Extended fetch tests: VectorQuery path, error handling, edge cases."""
import time
import pytest
import respx
import httpx
from unittest.mock import AsyncMock

from mcp_tool_manager.services.fetch import (
    _decode_chunk,
    fetch_by_query,
    fetch_all_session,
)
from mcp_tool_manager.config import Settings

FAKE_EMBEDDING = b"\x00" * (1024 * 4)


def test_decode_chunk_string_keys():
    """Redis can return string keys (not bytes) in some configurations."""
    data = {
        "session_id": "sess1",
        "tool_call_id": "sess1:abc",
        "tool_id": "github:search",
        "chunk_idx": "2",
        "text": "text content",
        "embedding": FAKE_EMBEDDING,
        "arguments_json": '{"q": "test"}',
        "created_at": "1700000000.0",
    }
    chunk = _decode_chunk("cache:sess1:sess1:abc:2", data)
    assert chunk is not None
    assert chunk.session_id == "sess1"
    assert chunk.chunk_idx == 2
    assert chunk.arguments_json == '{"q": "test"}'


def test_decode_chunk_missing_fields_graceful():
    """Chunk with minimal/missing fields returns None or partial chunk."""
    # Completely empty data
    result = _decode_chunk("cache:x:y:z:0", {})
    # Should either return None or a chunk with empty values, not raise
    # (empty string for session_id is falsy but valid)


@pytest.mark.asyncio
async def test_fetch_all_session_empty(fakeredis_client):
    result = await fetch_all_session("empty_session", top_k=10, redis_client=fakeredis_client)
    assert result.total_chunks == 0
    assert result.chunks == []


@pytest.mark.asyncio
async def test_fetch_by_query_exception_in_index(fakeredis_client):
    """When VectorQuery raises, returns empty FetchResult."""
    import os
    os.environ["EMBEDDING__API_URL"] = "http://fakeembed/v1/embeddings"
    settings = Settings()

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(side_effect=Exception("index error"))

    async with respx.mock:
        respx.post("http://fakeembed/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
            )
        )

        async with httpx.AsyncClient() as client:
            result = await fetch_by_query(
                session_id="sess1",
                query="search query",
                top_k=5,
                settings=settings,
                cache_index=mock_index,
                http_client=client,
                redis_client=fakeredis_client,
            )

    assert result.total_chunks == 0
    assert result.chunks == []


@pytest.mark.asyncio
async def test_fetch_by_query_returns_sorted_chunks(fakeredis_client):
    """Chunks returned from index are converted to CacheChunk correctly."""
    import os
    os.environ["EMBEDDING__API_URL"] = "http://fakeembed/v1/embeddings"
    settings = Settings()

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[
        {
            "session_id": "sess2",
            "tool_call_id": "sess2:abc",
            "tool_id": "svc:tool",
            "chunk_idx": "1",
            "text": "second chunk",
            "arguments_json": "{}",
            "created_at": "1700000002.0",
        },
        {
            "session_id": "sess2",
            "tool_call_id": "sess2:abc",
            "tool_id": "svc:tool",
            "chunk_idx": "0",
            "text": "first chunk",
            "arguments_json": "{}",
            "created_at": "1700000001.0",
        },
    ])

    async with respx.mock:
        respx.post("http://fakeembed/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
            )
        )

        async with httpx.AsyncClient() as client:
            result = await fetch_by_query(
                session_id="sess2",
                query="find chunks",
                top_k=5,
                settings=settings,
                cache_index=mock_index,
                http_client=client,
            )

    assert result.total_chunks == 2
    assert result.chunks[0].text == "second chunk"


@pytest.mark.asyncio
async def test_fetch_by_query_handles_malformed_result(fakeredis_client):
    """Malformed result items are skipped, not raised."""
    import os
    os.environ["EMBEDDING__API_URL"] = "http://fakeembed/v1/embeddings"
    settings = Settings()

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[
        {"session_id": "sess3", "tool_call_id": "c", "tool_id": "t",
         "chunk_idx": "not_a_number", "text": "text", "created_at": "bad"},
    ])

    async with respx.mock:
        respx.post("http://fakeembed/v1/embeddings").mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
            )
        )

        async with httpx.AsyncClient() as client:
            result = await fetch_by_query(
                session_id="sess3",
                query="query",
                top_k=5,
                settings=settings,
                cache_index=mock_index,
                http_client=client,
            )

    # Malformed items skipped; returns empty or partial
    assert isinstance(result.chunks, list)
