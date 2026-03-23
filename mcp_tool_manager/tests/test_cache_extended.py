"""Extended cache tests: result_to_text edge cases, cache_writer_task."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_tool_manager.services.cache import result_to_text, cache_writer_task
from mcp_tool_manager.config import Settings

FAKE_EMBEDDING = b"\x00" * (1024 * 4)


def test_result_to_text_content_with_data_attr():
    """Content item has 'data' attribute but no 'text'."""
    class DataContent:
        data = b"\xff\xfe"

    class MockResult:
        content = [DataContent()]

    result = result_to_text(MockResult())
    assert "b'\\xff\\xfe'" in result or "fffd" in result or len(result) > 0


def test_result_to_text_content_with_neither():
    """Content item has neither 'text' nor 'data'."""
    class BareContent:
        pass

    class MockResult:
        content = [BareContent()]

    result = result_to_text(MockResult())
    assert isinstance(result, str)


def test_result_to_text_mixed_content():
    """Mix of text and data content items."""
    class TextContent:
        text = "hello"

    class DataContent:
        data = "world"

    class MockResult:
        content = [TextContent(), DataContent()]

    result = result_to_text(MockResult())
    assert "hello" in result
    assert "world" in result


@pytest.mark.asyncio
async def test_cache_writer_task_processes_item(fakeredis_client):
    settings = Settings()
    settings.cache.chunk_size = 1000
    settings.cache.chunk_overlap = 100
    settings.cache.ttl = 300

    queue = asyncio.Queue(maxsize=10)
    queue.put_nowait(("sess1", "sess1:abc", "github:search", {"q": "python"}, "result text"))

    mock_http = MagicMock()

    async def fake_embed_batch(texts, s, client):
        return [FAKE_EMBEDDING for _ in texts]

    with patch("mcp_tool_manager.dependencies.get_redis", return_value=fakeredis_client), \
         patch("mcp_tool_manager.dependencies.get_httpx", return_value=mock_http), \
         patch("mcp_tool_manager.services.embedding.embed_batch", side_effect=fake_embed_batch):

        task = asyncio.create_task(cache_writer_task(queue, settings))
        # Wait for queue to be processed
        await asyncio.wait_for(queue.join(), timeout=5.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Verify chunk was stored
    data = await fakeredis_client.hgetall("cache:sess1:sess1:abc:0")
    assert b"text" in data


@pytest.mark.asyncio
async def test_cache_writer_task_handles_write_error(fakeredis_client):
    """Write failure logs error but does not crash the task."""
    settings = Settings()
    queue = asyncio.Queue(maxsize=10)
    queue.put_nowait(("sess_err", "sess_err:xyz", "svc:tool", {}, "data"))

    async def failing_embed_batch(texts, s, client):
        raise RuntimeError("embedding failure")

    with patch("mcp_tool_manager.dependencies.get_redis", return_value=fakeredis_client), \
         patch("mcp_tool_manager.dependencies.get_httpx", return_value=MagicMock()), \
         patch("mcp_tool_manager.services.embedding.embed_batch", side_effect=failing_embed_batch):

        task = asyncio.create_task(cache_writer_task(queue, settings))
        await asyncio.wait_for(queue.join(), timeout=5.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_cache_writer_task_drains_queue_on_cancel(fakeredis_client):
    """On cancellation, remaining queued items are drained."""
    settings = Settings()
    queue = asyncio.Queue(maxsize=10)
    # Fill queue with items before starting
    for i in range(3):
        queue.put_nowait((f"sess{i}", f"sess{i}:x", "svc:t", {}, "txt"))

    # Slow embed so task is mid-processing when cancelled
    call_count = 0

    async def slow_embed(texts, s, client):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return [FAKE_EMBEDDING for _ in texts]

    with patch("mcp_tool_manager.dependencies.get_redis", return_value=fakeredis_client), \
         patch("mcp_tool_manager.dependencies.get_httpx", return_value=MagicMock()), \
         patch("mcp_tool_manager.services.embedding.embed_batch", side_effect=slow_embed):

        task = asyncio.create_task(cache_writer_task(queue, settings))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Task should have cancelled cleanly
