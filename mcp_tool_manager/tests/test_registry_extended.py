"""Extended registry tests: error paths, force sync, delete, embedding text."""
import json
import pytest
import respx
import httpx
from unittest.mock import AsyncMock

from mcp_tool_manager.services.registry import (
    run_sync_job,
    get_sync_job_status,
    _compose_embedding_text,
    _content_hash,
)
from mcp_tool_manager.models import ToolDoc
from mcp_tool_manager.config import Settings

FAKE_VECTOR_BYTES = b"\x00" * (1024 * 4)


def make_settings():
    settings = Settings()
    settings.registry.api_url = "http://fakeregistry"
    return settings


SERVERS_RESPONSE = {"servers": [
    {"server_name": "github", "description": "GitHub tools", "url": "http://github-mcp"},
]}

GITHUB_TOOLS_RESPONSE = {"tools": [{
    "name": "search_repos",
    "description": "Search GitHub repositories",
    "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
    "tags": ["search"],
}]}


@pytest.mark.asyncio
async def test_run_sync_job_force_re_embeds_unchanged(fakeredis_client):
    """force=True re-embeds even if hash unchanged."""
    settings = make_settings()

    # Pre-populate with same hash
    chash = _content_hash("github", "search_repos", "Search GitHub repositories",
                          {"type": "object", "properties": {"query": {"type": "string"}}})
    await fakeredis_client.hset("tool:github:search_repos", "content_hash", chash)

    embed_call_count = 0

    async def counting_embed_batch(texts):
        nonlocal embed_call_count
        embed_call_count += len(texts)
        return [FAKE_VECTOR_BYTES for _ in texts]

    async with respx.mock:
        respx.get("http://fakeregistry/mcp/servers").mock(
            return_value=httpx.Response(200, json=SERVERS_RESPONSE)
        )
        respx.get("http://fakeregistry/mcp/servers/github/tools").mock(
            return_value=httpx.Response(200, json=GITHUB_TOOLS_RESPONSE)
        )

        async with httpx.AsyncClient() as http_client:
            await run_sync_job(
                job_id="sync_force01",
                force=True,
                settings=settings,
                http_client=http_client,
                redis_client=fakeredis_client,
                embed_batch_fn=counting_embed_batch,
            )

    job = await get_sync_job_status("sync_force01", fakeredis_client)
    assert job.status == "completed"
    assert embed_call_count == 1  # Force re-embedded despite unchanged hash
    assert job.stats.updated == 1  # Counted as update (hash existed before)


@pytest.mark.asyncio
async def test_run_sync_job_deletes_stale_tools(fakeredis_client):
    """Tools in Redis but not in registry are deleted."""
    settings = make_settings()

    # Pre-populate a stale tool that won't be returned by registry
    await fakeredis_client.hset("tool:old_server:stale_tool", "content_hash", "old_hash")
    await fakeredis_client.hset("tool:old_server:another_stale", "content_hash", "old_hash2")

    async def fake_embed_batch(texts):
        return [FAKE_VECTOR_BYTES for _ in texts]

    async with respx.mock:
        respx.get("http://fakeregistry/mcp/servers").mock(
            return_value=httpx.Response(200, json=SERVERS_RESPONSE)
        )
        respx.get("http://fakeregistry/mcp/servers/github/tools").mock(
            return_value=httpx.Response(200, json=GITHUB_TOOLS_RESPONSE)
        )

        async with httpx.AsyncClient() as http_client:
            await run_sync_job(
                job_id="sync_del01",
                force=False,
                settings=settings,
                http_client=http_client,
                redis_client=fakeredis_client,
                embed_batch_fn=fake_embed_batch,
            )

    job = await get_sync_job_status("sync_del01", fakeredis_client)
    assert job.status == "completed"
    assert job.stats.deleted == 2

    # Verify stale keys are gone
    assert await fakeredis_client.exists("tool:old_server:stale_tool") == 0
    assert await fakeredis_client.exists("tool:old_server:another_stale") == 0


@pytest.mark.asyncio
async def test_run_sync_job_fails_on_registry_error(fakeredis_client):
    """Registry fetch failure transitions job to failed status."""
    settings = make_settings()

    async def fake_embed_batch(texts):
        return []

    async with respx.mock:
        respx.get("http://fakeregistry/mcp/servers").mock(
            return_value=httpx.Response(500)
        )

        async with httpx.AsyncClient() as http_client:
            await run_sync_job(
                job_id="sync_fail01",
                force=False,
                settings=settings,
                http_client=http_client,
                redis_client=fakeredis_client,
                embed_batch_fn=fake_embed_batch,
            )

    job = await get_sync_job_status("sync_fail01", fakeredis_client)
    assert job.status == "failed"
    assert job.error is not None
    assert job.completed_at is not None


@pytest.mark.asyncio
async def test_run_sync_job_empty_registry(fakeredis_client):
    """Empty registry completes successfully with zero stats."""
    settings = make_settings()

    async def fake_embed_batch(texts):
        return []

    async with respx.mock:
        respx.get("http://fakeregistry/mcp/servers").mock(
            return_value=httpx.Response(200, json={"servers": []})
        )

        async with httpx.AsyncClient() as http_client:
            await run_sync_job(
                job_id="sync_empty01",
                force=False,
                settings=settings,
                http_client=http_client,
                redis_client=fakeredis_client,
                embed_batch_fn=fake_embed_batch,
            )

    job = await get_sync_job_status("sync_empty01", fakeredis_client)
    assert job.status == "completed"
    assert job.stats.fetched == 0


def test_compose_embedding_text_no_params():
    """Tool with no parameters still produces valid embedding text."""
    doc = ToolDoc(
        tool_id="svc:tool",
        name="tool",
        description="A simple tool",
        server_name="svc",
        server_description="",
        tags=[],
        input_schema={},
    )
    text = _compose_embedding_text(doc)
    assert "tool" in text
    assert "A simple tool" in text


def test_compose_embedding_text_empty_param_description():
    """Param with no description just includes name."""
    doc = ToolDoc(
        tool_id="svc:tool",
        name="tool",
        description="desc",
        server_name="svc",
        input_schema={"properties": {"param1": {"type": "string"}}},
    )
    text = _compose_embedding_text(doc)
    assert "param1" in text


@pytest.mark.asyncio
async def test_run_sync_job_lock_released_on_success(fakeredis_client):
    """Lock key is deleted after sync completes."""
    settings = make_settings()
    await fakeredis_client.set("sync:lock", "sync_lock01", ex=300)

    async def fake_embed_batch(texts):
        return [FAKE_VECTOR_BYTES for _ in texts]

    async with respx.mock:
        respx.get("http://fakeregistry/mcp/servers").mock(
            return_value=httpx.Response(200, json={"servers": []})
        )

        async with httpx.AsyncClient() as http_client:
            await run_sync_job(
                job_id="sync_lock01",
                force=False,
                settings=settings,
                http_client=http_client,
                redis_client=fakeredis_client,
                embed_batch_fn=fake_embed_batch,
            )

    assert await fakeredis_client.exists("sync:lock") == 0


@pytest.mark.asyncio
async def test_run_sync_job_lock_released_on_failure(fakeredis_client):
    """Lock key is deleted even when sync fails."""
    settings = make_settings()
    await fakeredis_client.set("sync:lock", "sync_faillock01", ex=300)

    async def fake_embed_batch(texts):
        return []

    async with respx.mock:
        respx.get("http://fakeregistry/mcp/servers").mock(
            return_value=httpx.Response(500)
        )

        async with httpx.AsyncClient() as http_client:
            await run_sync_job(
                job_id="sync_faillock01",
                force=False,
                settings=settings,
                http_client=http_client,
                redis_client=fakeredis_client,
                embed_batch_fn=fake_embed_batch,
            )

    assert await fakeredis_client.exists("sync:lock") == 0
