"""Tests for vLLM embedding client."""
import struct
import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock

from mcp_tool_manager.services.embedding import (
    embed,
    embed_batch,
    _floats_to_bytes,
    _bytes_to_floats,
    _query_cache_key,
)
from mcp_tool_manager.config import Settings


FAKE_VECTOR = [0.1] * 1024


def make_settings(**kwargs):
    overrides = {
        "EMBEDDING__API_URL": "http://fakehost/v1/embeddings",
        "EMBEDDING__MODEL": "test-model",
        "EMBEDDING__BATCH_SIZE": "32",
    }
    overrides.update(kwargs)
    import os
    old = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        os.environ[k] = v
    settings = Settings()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return settings


def test_floats_roundtrip():
    floats = [0.1, 0.2, 0.3, 0.5, 1.0]
    encoded = _floats_to_bytes(floats)
    decoded = _bytes_to_floats(encoded)
    assert len(decoded) == 5
    for a, b in zip(floats, decoded):
        assert abs(a - b) < 1e-5


def test_query_cache_key_stable():
    key1 = _query_cache_key("search github repos")
    key2 = _query_cache_key("search github repos")
    key3 = _query_cache_key("different query")
    assert key1 == key2
    assert key1 != key3
    assert key1.startswith("qemb:")


@respx.mock
@pytest.mark.asyncio
async def test_embed_single():
    settings = make_settings()

    respx.post("http://fakehost/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": FAKE_VECTOR, "index": 0}]},
        )
    )

    async with httpx.AsyncClient() as client:
        result = await embed("test query", settings, client)

    assert isinstance(result, bytes)
    decoded = _bytes_to_floats(result)
    assert len(decoded) == 1024
    assert abs(decoded[0] - 0.1) < 1e-5


@respx.mock
@pytest.mark.asyncio
async def test_embed_uses_cache(fakeredis_client):
    settings = make_settings()
    cache_key = _query_cache_key("cached query")

    # Pre-populate cache
    cached_bytes = _floats_to_bytes(FAKE_VECTOR)
    await fakeredis_client.setex(cache_key, 300, cached_bytes)

    # No HTTP call should be made
    async with httpx.AsyncClient() as client:
        result = await embed("cached query", settings, client, redis_client=fakeredis_client)

    assert result == cached_bytes


@respx.mock
@pytest.mark.asyncio
async def test_embed_stores_in_cache(fakeredis_client):
    settings = make_settings()

    respx.post("http://fakehost/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": FAKE_VECTOR, "index": 0}]},
        )
    )

    async with httpx.AsyncClient() as client:
        result = await embed("new query", settings, client, redis_client=fakeredis_client)

    cache_key = _query_cache_key("new query")
    cached = await fakeredis_client.get(cache_key)
    assert cached == result


@respx.mock
@pytest.mark.asyncio
async def test_embed_batch():
    settings = make_settings()
    texts = ["text one", "text two", "text three"]
    batch_response = {"data": [{"embedding": FAKE_VECTOR, "index": i} for i in range(3)]}

    respx.post("http://fakehost/v1/embeddings").mock(
        return_value=httpx.Response(200, json=batch_response)
    )

    async with httpx.AsyncClient() as client:
        results = await embed_batch(texts, settings, client)

    assert len(results) == 3
    for r in results:
        assert isinstance(r, bytes)
        assert len(_bytes_to_floats(r)) == 1024


@pytest.mark.asyncio
async def test_embed_batch_empty():
    settings = make_settings()
    async with httpx.AsyncClient() as client:
        results = await embed_batch([], settings, client)
    assert results == []
