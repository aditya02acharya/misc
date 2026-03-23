"""vLLM embedding client: embed single query or batch of texts."""
import hashlib
import logging
import struct
import time
from typing import Any

import httpx

from mcp_tool_manager.telemetry import record_histogram

logger = logging.getLogger(__name__)

# Redis key pattern for cached query embeddings (TTL 300s)
QUERY_EMB_TTL = 300


def _query_cache_key(text: str) -> str:
    return f"qemb:{hashlib.sha256(text.encode()).hexdigest()[:16]}"


def _floats_to_bytes(floats: list[float]) -> bytes:
    return struct.pack(f"{len(floats)}f", *floats)


def _bytes_to_floats(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


async def embed(
    text: str,
    settings: Any,
    http_client: httpx.AsyncClient,
    redis_client: Any = None,
) -> bytes:
    """Embed a single text. Caches in Redis if redis_client provided."""
    if redis_client is not None:
        cache_key = _query_cache_key(text)
        cached = await redis_client.get(cache_key)
        if cached:
            logger.debug("Query embedding cache hit: %s", cache_key)
            return cached

    vector = await _call_embedding_api(text, settings, http_client)
    result = _floats_to_bytes(vector)

    if redis_client is not None:
        await redis_client.setex(cache_key, QUERY_EMB_TTL, result)

    return result


async def embed_batch(
    texts: list[str],
    settings: Any,
    http_client: httpx.AsyncClient,
) -> list[bytes]:
    """Embed a batch of texts. Returns list of float-packed bytes."""
    if not texts:
        return []

    results = []
    batch_size = settings.embedding.batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await http_client.post(
            settings.embedding.api_url,
            json={"model": settings.embedding.model, "input": batch},
        )
        response.raise_for_status()
        data = response.json()
        for item in data["data"]:
            results.append(_floats_to_bytes(item["embedding"]))

    return results


async def _call_embedding_api(
    text: str,
    settings: Any,
    http_client: httpx.AsyncClient,
) -> list[float]:
    start = time.monotonic()
    response = await http_client.post(
        settings.embedding.api_url,
        json={"model": settings.embedding.model, "input": [text]},
    )
    response.raise_for_status()
    record_histogram("mcp.embedding.latency_ms", (time.monotonic() - start) * 1000)
    data = response.json()
    return data["data"][0]["embedding"]
