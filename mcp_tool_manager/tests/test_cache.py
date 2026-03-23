"""Tests for cache service: chunking, serialization, Redis writes."""
import json
import pytest
import respx
import httpx
from unittest.mock import AsyncMock

from mcp_tool_manager.services.cache import (
    chunk_text,
    result_to_text,
    write_cache_chunks,
)
from mcp_tool_manager.config import Settings


FAKE_EMBEDDING = b"\x00" * (1024 * 4)


def make_settings():
    settings = Settings()
    settings.cache.chunk_size = 100
    settings.cache.chunk_overlap = 20
    settings.cache.ttl = 3600
    return settings


def test_chunk_text_short():
    chunks = chunk_text("short text", 1000, 100)
    assert chunks == ["short text"]


def test_chunk_text_empty():
    assert chunk_text("", 100, 20) == []


def test_chunk_text_exact_size():
    text = "a" * 100
    chunks = chunk_text(text, 100, 20)
    assert chunks == [text]


def test_chunk_text_splits_with_overlap():
    text = "a" * 200
    chunks = chunk_text(text, 100, 20)
    # First chunk: [0:100], second: [80:180], third: [160:200]
    assert len(chunks) == 3
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100


def test_chunk_text_overlap_content():
    text = "0123456789" * 20  # 200 chars
    chunks = chunk_text(text, 100, 20)
    # Overlap: last 20 chars of chunk[0] should equal first 20 chars of chunk[1]
    assert chunks[0][-20:] == chunks[1][:20]


def test_result_to_text_string():
    assert result_to_text("hello world") == "hello world"


def test_result_to_text_dict():
    result = result_to_text({"key": "value", "count": 5})
    assert "key" in result
    assert "value" in result


def test_result_to_text_mcp_result():
    """Test with MCP-like result object with .content attribute."""
    class MockContent:
        text = "tool output"

    class MockResult:
        content = [MockContent()]

    result = result_to_text(MockResult())
    assert result == "tool output"


def test_result_to_text_none():
    result = result_to_text(None)
    assert result == "None"


@pytest.mark.asyncio
async def test_write_cache_chunks_single(fakeredis_client):
    settings = make_settings()

    async def fake_embed_batch(texts):
        return [FAKE_EMBEDDING for _ in texts]

    count = await write_cache_chunks(
        session_id="sess1",
        tool_call_id="sess1:abc12345",
        tool_id="github:search_repos",
        arguments={"query": "python"},
        raw_result="Short result text",
        redis_client=fakeredis_client,
        settings=settings,
        embed_batch_fn=fake_embed_batch,
    )

    assert count == 1
    # Verify stored in Redis
    data = await fakeredis_client.hgetall("cache:sess1:sess1:abc12345:0")
    assert b"text" in data
    assert b"Short result text" in data[b"text"]
    assert b"session_id" in data


@pytest.mark.asyncio
async def test_write_cache_chunks_multiple(fakeredis_client):
    settings = make_settings()
    settings.cache.chunk_size = 50
    settings.cache.chunk_overlap = 10

    async def fake_embed_batch(texts):
        return [FAKE_EMBEDDING for _ in texts]

    long_text = "x" * 200
    count = await write_cache_chunks(
        session_id="sess2",
        tool_call_id="sess2:def67890",
        tool_id="slack:send_message",
        arguments={},
        raw_result=long_text,
        redis_client=fakeredis_client,
        settings=settings,
        embed_batch_fn=fake_embed_batch,
    )

    assert count > 1
    # All chunks stored
    for i in range(count):
        data = await fakeredis_client.hgetall(f"cache:sess2:sess2:def67890:{i}")
        assert b"text" in data


@pytest.mark.asyncio
async def test_write_cache_chunks_has_ttl(fakeredis_client):
    settings = make_settings()
    settings.cache.ttl = 3600

    async def fake_embed_batch(texts):
        return [FAKE_EMBEDDING for _ in texts]

    await write_cache_chunks(
        session_id="sess3",
        tool_call_id="sess3:aaa",
        tool_id="test:tool",
        arguments={},
        raw_result="some text",
        redis_client=fakeredis_client,
        settings=settings,
        embed_batch_fn=fake_embed_batch,
    )

    ttl = await fakeredis_client.ttl("cache:sess3:sess3:aaa:0")
    assert ttl > 0


@pytest.mark.asyncio
async def test_write_cache_chunks_empty_result(fakeredis_client):
    settings = make_settings()

    async def fake_embed_batch(texts):
        return [FAKE_EMBEDDING for _ in texts]

    count = await write_cache_chunks(
        session_id="sess4",
        tool_call_id="sess4:bbb",
        tool_id="test:tool",
        arguments={},
        raw_result="",
        redis_client=fakeredis_client,
        settings=settings,
        embed_batch_fn=fake_embed_batch,
    )

    assert count == 0
