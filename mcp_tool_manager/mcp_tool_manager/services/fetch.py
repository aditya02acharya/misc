"""Cache retrieval service: all three fetch modes."""
import logging
from typing import Any

from mcp_tool_manager.models import CacheChunk, FetchResult

logger = logging.getLogger(__name__)


def _decode_chunk(key: str, data: dict) -> CacheChunk | None:
    """Build a CacheChunk from Redis hash data."""
    try:
        def _d(v):
            return v.decode() if isinstance(v, bytes) else (v or "")

        return CacheChunk(
            session_id=_d(data.get(b"session_id") or data.get("session_id")),
            tool_call_id=_d(data.get(b"tool_call_id") or data.get("tool_call_id")),
            tool_id=_d(data.get(b"tool_id") or data.get("tool_id")),
            chunk_idx=int(_d(data.get(b"chunk_idx") or data.get("chunk_idx") or "0")),
            text=_d(data.get(b"text") or data.get("text")),
            embedding=data.get(b"embedding") or data.get("embedding") or b"",
            arguments_json=_d(data.get(b"arguments_json") or data.get("arguments_json") or "{}"),
            created_at=float(
                _d(data.get(b"created_at") or data.get("created_at") or "0")
            ),
        )
    except Exception as exc:
        logger.warning("Failed to decode chunk from key %s: %s", key, exc)
        return None


async def fetch_by_tool_call_id(
    session_id: str,
    tool_call_id: str,
    redis_client: Any,
) -> FetchResult:
    """Case 1: fetch all chunks for a specific tool_call_id."""
    pattern = f"cache:{session_id}:{tool_call_id}:*"
    chunks = []
    cursor = 0

    while True:
        cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            data = await redis_client.hgetall(key_str)
            chunk = _decode_chunk(key_str, data)
            if chunk:
                chunks.append(chunk)
        if cursor == 0:
            break

    # Sort by chunk_idx
    chunks.sort(key=lambda c: c.chunk_idx)
    return FetchResult(chunks=chunks, total_chunks=len(chunks))


async def fetch_by_query(
    session_id: str,
    query: str,
    top_k: int,
    settings: Any,
    cache_index: Any,
    http_client: Any,
    redis_client: Any = None,
) -> FetchResult:
    """Case 2: semantic search within session cache."""
    from mcp_tool_manager.services.embedding import embed

    if cache_index is None:
        logger.warning("Cache index not available")
        return FetchResult(chunks=[], total_chunks=0)

    query_vector = await embed(query, settings, http_client, redis_client)

    try:
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        vq = VectorQuery(
            vector=query_vector,
            vector_field_name="embedding",
            return_fields=[
                "session_id",
                "tool_call_id",
                "tool_id",
                "chunk_idx",
                "text",
                "arguments_json",
                "created_at",
            ],
            filter_expression=Tag("session_id") == session_id,
            num_results=top_k,
        )
        results = await cache_index.query(vq)

    except Exception as exc:
        logger.error("Cache semantic search failed: %s", exc)
        return FetchResult(chunks=[], total_chunks=0)

    chunks = []
    for item in results or []:
        try:
            chunks.append(
                CacheChunk(
                    session_id=item.get("session_id", ""),
                    tool_call_id=item.get("tool_call_id", ""),
                    tool_id=item.get("tool_id", ""),
                    chunk_idx=int(item.get("chunk_idx", 0)),
                    text=item.get("text", ""),
                    arguments_json=item.get("arguments_json", "{}"),
                    created_at=float(item.get("created_at", 0)),
                )
            )
        except Exception as exc:
            logger.warning("Failed to parse cache result: %s", exc)

    return FetchResult(chunks=chunks, total_chunks=len(chunks))


async def fetch_all_session(
    session_id: str,
    top_k: int,
    redis_client: Any,
) -> FetchResult:
    """Case 3: all chunks for a session, sorted by created_at desc."""
    pattern = f"cache:{session_id}:*"
    chunks = []
    cursor = 0

    while True:
        cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            data = await redis_client.hgetall(key_str)
            chunk = _decode_chunk(key_str, data)
            if chunk:
                chunks.append(chunk)
        if cursor == 0:
            break

    # Sort by created_at desc (most recent first)
    chunks.sort(key=lambda c: c.created_at, reverse=True)

    # Paginate with top_k
    paginated = chunks[:top_k] if top_k else chunks
    return FetchResult(chunks=paginated, total_chunks=len(chunks))
