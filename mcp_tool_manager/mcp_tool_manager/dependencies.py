"""Shared singleton holders for Redis, HTTP client, Bedrock client, and async queues."""
import asyncio
import logging
from typing import Any

import boto3
import httpx
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Singletons
_redis_client: Redis | None = None
_httpx_client: httpx.AsyncClient | None = None
_bedrock_client: Any | None = None
_cache_queue: asyncio.Queue | None = None

# Tool index and cache index (redisvl SearchIndex instances)
_tool_index: Any | None = None
_cache_index: Any | None = None


def get_redis() -> Redis:
    if _redis_client is None:
        raise RuntimeError("Redis client not initialized. Call init_dependencies() first.")
    return _redis_client


def get_httpx() -> httpx.AsyncClient:
    if _httpx_client is None:
        raise RuntimeError("HTTPX client not initialized. Call init_dependencies() first.")
    return _httpx_client


def get_bedrock() -> Any:
    if _bedrock_client is None:
        raise RuntimeError("Bedrock client not initialized. Call init_dependencies() first.")
    return _bedrock_client


def get_cache_queue() -> asyncio.Queue:
    if _cache_queue is None:
        raise RuntimeError("Cache queue not initialized. Call init_dependencies() first.")
    return _cache_queue


def get_tool_index() -> Any:
    return _tool_index


def get_cache_index() -> Any:
    return _cache_index


async def init_dependencies(settings) -> None:
    global _redis_client, _httpx_client, _bedrock_client, _cache_queue

    # Redis async client
    _redis_client = Redis.from_url(settings.redis.url, decode_responses=False)
    logger.info("Redis client initialized: %s", settings.redis.url)

    # HTTPX async client for vLLM embeddings
    headers = {}
    if settings.embedding.api_key:
        headers["Authorization"] = f"Bearer {settings.embedding.api_key}"
    _httpx_client = httpx.AsyncClient(
        timeout=settings.embedding.timeout,
        headers=headers,
    )
    logger.info("HTTPX client initialized")

    # Boto3 Bedrock client (sync, wrapped in asyncio.to_thread at call site)
    kwargs: dict = {"region_name": settings.bedrock.region}
    if settings.bedrock.endpoint_url:
        kwargs["endpoint_url"] = settings.bedrock.endpoint_url
    _bedrock_client = boto3.client("bedrock-runtime", **kwargs)
    logger.info("Bedrock client initialized (region=%s)", settings.bedrock.region)

    # Async queue for background cache writes
    _cache_queue = asyncio.Queue(maxsize=100)
    logger.info("Cache queue initialized")


async def init_redis_indexes(settings) -> None:
    """Create redisvl search indexes if they don't exist."""
    global _tool_index, _cache_index

    from redisvl.index import SearchIndex

    tool_schema = {
        "index": {
            "name": settings.redis.tool_index,
            "prefix": "tool:",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "name", "type": "text", "attrs": {"weight": 3.0}},
            {"name": "name_tag", "type": "tag", "attrs": {"separator": ","}},
            {"name": "description", "type": "text", "attrs": {"weight": 2.0}},
            {"name": "searchable_text", "type": "text", "attrs": {"weight": 1.0}},
            {"name": "server_name", "type": "tag", "attrs": {"separator": ","}},
            {"name": "tags", "type": "tag", "attrs": {"separator": ","}},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",
                    "dims": settings.redis.vector_dims,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
            {"name": "input_schema", "type": "text", "attrs": {"no_index": True}},
            {"name": "content_hash", "type": "text", "attrs": {"no_index": True}},
        ],
    }

    cache_schema = {
        "index": {
            "name": settings.redis.cache_index,
            "prefix": "cache:",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "session_id", "type": "tag"},
            {"name": "tool_call_id", "type": "tag"},
            {"name": "tool_id", "type": "tag"},
            {"name": "text", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "hnsw",
                    "dims": settings.redis.vector_dims,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
            {"name": "created_at", "type": "numeric", "attrs": {"sortable": True}},
        ],
    }

    redis_url = settings.redis.url
    _tool_index = SearchIndex.from_dict(tool_schema)
    await _tool_index.set_client(get_redis())
    await _tool_index.create(overwrite=False)
    logger.info("Tool search index ready: %s", settings.redis.tool_index)

    _cache_index = SearchIndex.from_dict(cache_schema)
    await _cache_index.set_client(get_redis())
    await _cache_index.create(overwrite=False)
    logger.info("Cache search index ready: %s", settings.redis.cache_index)


async def close_dependencies() -> None:
    global _redis_client, _httpx_client

    if _httpx_client:
        await _httpx_client.aclose()
        logger.info("HTTPX client closed")

    if _redis_client:
        await _redis_client.aclose()
        logger.info("Redis client closed")
