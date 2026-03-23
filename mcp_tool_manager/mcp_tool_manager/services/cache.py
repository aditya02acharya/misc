"""Async cache writer: chunk → embed → store in Redis with TTL."""
import asyncio
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap

    return chunks


def result_to_text(raw_result: Any) -> str:
    """Serialize MCP tool result to plain text."""
    if isinstance(raw_result, str):
        return raw_result

    if hasattr(raw_result, "content"):
        # MCP CallToolResult type
        parts = []
        for content_item in raw_result.content:
            if hasattr(content_item, "text"):
                parts.append(content_item.text)
            elif hasattr(content_item, "data"):
                parts.append(str(content_item.data))
            else:
                parts.append(str(content_item))
        return "\n".join(parts)

    if isinstance(raw_result, dict):
        return json.dumps(raw_result, indent=2)

    return str(raw_result)


async def write_cache_chunks(
    session_id: str,
    tool_call_id: str,
    tool_id: str,
    arguments: dict,
    raw_result: Any,
    redis_client: Any,
    settings: Any,
    embed_batch_fn,
) -> int:
    """Chunk, embed, and store result in Redis. Returns chunk count."""
    text = result_to_text(raw_result)
    chunks = chunk_text(text, settings.cache.chunk_size, settings.cache.chunk_overlap)

    if not chunks:
        return 0

    # Batch embed all chunks
    embeddings = await embed_batch_fn(chunks)
    now = time.time()
    args_json = json.dumps(arguments, default=str)

    pipe = redis_client.pipeline()
    for idx, (chunk_text_, embedding) in enumerate(zip(chunks, embeddings)):
        key = f"cache:{session_id}:{tool_call_id}:{idx}"
        data = {
            "session_id": session_id,
            "tool_call_id": tool_call_id,
            "tool_id": tool_id,
            "chunk_idx": str(idx),
            "text": chunk_text_,
            "embedding": embedding,
            "arguments_json": args_json,
            "created_at": str(now),
        }
        pipe.hset(key, mapping=data)
        pipe.expire(key, settings.cache.ttl)

    await pipe.execute()
    return len(chunks)


async def cache_writer_task(queue: asyncio.Queue, settings: Any) -> None:
    """
    Background task: consume from queue and write cache chunks.
    Runs forever until cancelled.
    """
    from mcp_tool_manager.dependencies import get_redis, get_httpx
    from mcp_tool_manager.services.embedding import embed_batch

    logger.info("Cache writer task started")

    while True:
        try:
            item = await queue.get()
            session_id, tool_call_id, tool_id, arguments, raw_result = item

            redis_client = get_redis()
            http_client = get_httpx()

            async def _embed_batch(texts):
                return await embed_batch(texts, settings, http_client)

            try:
                count = await write_cache_chunks(
                    session_id=session_id,
                    tool_call_id=tool_call_id,
                    tool_id=tool_id,
                    arguments=arguments,
                    raw_result=raw_result,
                    redis_client=redis_client,
                    settings=settings,
                    embed_batch_fn=_embed_batch,
                )
                logger.debug("Cached %d chunks for %s", count, tool_call_id)
            except Exception as exc:
                logger.error("Cache write failed for %s: %s", tool_call_id, exc)
            finally:
                queue.task_done()

        except asyncio.CancelledError:
            logger.info("Cache writer task cancelled, draining queue")
            # Drain remaining items
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
            break
        except Exception as exc:
            logger.error("Unexpected error in cache writer: %s", exc)
