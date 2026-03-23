"""Tests for registry service: sync logic, diff, hashing."""
import json
import pytest
import respx
import httpx
from unittest.mock import AsyncMock

from mcp_tool_manager.services.registry import (
    _content_hash,
    _compose_embedding_text,
    fetch_servers,
    fetch_tools_for_server,
    run_sync_job,
    start_sync,
    get_sync_job_status,
)
from mcp_tool_manager.models import ToolDoc
from mcp_tool_manager.config import Settings


FAKE_VECTOR_BYTES = b"\x00" * (1024 * 4)


def make_settings():
    settings = Settings()
    settings.registry.api_url = "http://fakeregistry"
    settings.registry.api_key = ""
    return settings


SERVERS_RESPONSE = {
    "servers": [
        {"server_name": "github", "description": "GitHub tools", "url": "http://github-mcp"},
        {"server_name": "slack", "description": "Slack tools", "url": "http://slack-mcp"},
    ]
}

GITHUB_TOOLS_RESPONSE = {
    "tools": [
        {
            "name": "search_repos",
            "description": "Search GitHub repositories",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "language": {"type": "string", "description": "Programming language filter"},
                }
            },
            "tags": ["search", "github"],
        },
        {
            "name": "create_issue",
            "description": "Create a GitHub issue",
            "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}},
            "tags": ["write", "github"],
        },
    ]
}

SLACK_TOOLS_RESPONSE = {
    "tools": [
        {
            "name": "send_message",
            "description": "Send a Slack message",
            "inputSchema": {"type": "object", "properties": {"channel": {"type": "string"}}},
            "tags": ["messaging"],
        }
    ]
}


def test_content_hash_stable():
    h1 = _content_hash("github", "search_repos", "Search repos", {})
    h2 = _content_hash("github", "search_repos", "Search repos", {})
    assert h1 == h2
    assert len(h1) == 16


def test_content_hash_differs_on_change():
    h1 = _content_hash("github", "search_repos", "Search repos", {})
    h2 = _content_hash("github", "search_repos", "Search repos v2", {})
    assert h1 != h2


def test_compose_embedding_text():
    doc = ToolDoc(
        tool_id="github:search_repos",
        name="search_repos",
        description="Search GitHub repositories",
        server_name="github",
        server_description="GitHub tools server",
        tags=["search", "api"],
        input_schema={
            "properties": {
                "query": {"description": "Search query"},
                "language": {"description": "Language filter"},
            }
        },
    )
    text = _compose_embedding_text(doc)
    assert "search_repos" in text
    assert "Search GitHub repositories" in text
    assert "github" in text
    assert "search, api" in text
    assert "query" in text


def test_compose_embedding_text_truncates_params():
    """Only first 5 params included."""
    doc = ToolDoc(
        tool_id="svc:tool",
        name="tool",
        description="desc",
        server_name="svc",
        input_schema={
            "properties": {f"param{i}": {"description": f"desc{i}"} for i in range(10)}
        },
    )
    text = _compose_embedding_text(doc)
    # param0..param4 but not param5..param9
    assert "param4" in text
    assert "param5" not in text


@respx.mock
@pytest.mark.asyncio
async def test_fetch_servers():
    settings = make_settings()
    respx.get("http://fakeregistry/mcp/servers").mock(
        return_value=httpx.Response(200, json=SERVERS_RESPONSE)
    )

    async with httpx.AsyncClient() as client:
        servers = await fetch_servers(settings, client)

    assert len(servers) == 2
    assert servers[0]["server_name"] == "github"


@respx.mock
@pytest.mark.asyncio
async def test_fetch_tools_for_server():
    settings = make_settings()
    server = {"server_name": "github", "description": "GitHub tools"}
    respx.get("http://fakeregistry/mcp/servers/github/tools").mock(
        return_value=httpx.Response(200, json=GITHUB_TOOLS_RESPONSE)
    )

    async with httpx.AsyncClient() as client:
        tools = await fetch_tools_for_server(server, settings, client)

    assert len(tools) == 2
    assert tools[0].tool_id == "github:search_repos"
    assert tools[0].server_name == "github"
    assert "search" in tools[0].tags
    assert tools[0].content_hash != ""


@respx.mock
@pytest.mark.asyncio
async def test_fetch_tools_handles_failure():
    settings = make_settings()
    server = {"server_name": "broken", "description": "Broken server"}
    respx.get("http://fakeregistry/mcp/servers/broken/tools").mock(
        return_value=httpx.Response(500)
    )

    async with httpx.AsyncClient() as client:
        tools = await fetch_tools_for_server(server, settings, client)

    assert tools == []


@pytest.mark.asyncio
async def test_run_sync_job_creates_tools(fakeredis_client):
    settings = make_settings()
    settings.embedding.api_url = "http://fakeembed/v1/embeddings"

    async def fake_embed_batch(texts):
        return [FAKE_VECTOR_BYTES for _ in texts]

    async with respx.mock:
        respx.get("http://fakeregistry/mcp/servers").mock(
            return_value=httpx.Response(200, json=SERVERS_RESPONSE)
        )
        respx.get("http://fakeregistry/mcp/servers/github/tools").mock(
            return_value=httpx.Response(200, json=GITHUB_TOOLS_RESPONSE)
        )
        respx.get("http://fakeregistry/mcp/servers/slack/tools").mock(
            return_value=httpx.Response(200, json=SLACK_TOOLS_RESPONSE)
        )

        async with httpx.AsyncClient() as http_client:
            await run_sync_job(
                job_id="sync_test001",
                force=False,
                settings=settings,
                http_client=http_client,
                redis_client=fakeredis_client,
                embed_batch_fn=fake_embed_batch,
            )

    # Check job status
    job = await get_sync_job_status("sync_test001", fakeredis_client)
    assert job is not None
    assert job.status == "completed"
    assert job.stats.fetched == 3
    assert job.stats.created == 3
    assert job.stats.updated == 0
    assert job.stats.unchanged == 0

    # Check tools stored in Redis
    tool_data = await fakeredis_client.hgetall("tool:github:search_repos")
    assert tool_data is not None
    assert b"name" in tool_data


@pytest.mark.asyncio
async def test_run_sync_job_skips_unchanged(fakeredis_client):
    settings = make_settings()
    tool = ToolDoc(
        tool_id="github:search_repos",
        name="search_repos",
        description="Search GitHub repositories",
        server_name="github",
        tags=["search"],
        input_schema={},
    )
    from mcp_tool_manager.services.registry import _content_hash
    chash = _content_hash("github", "search_repos", "Search GitHub repositories", {})

    # Pre-populate with same hash
    await fakeredis_client.hset("tool:github:search_repos", "content_hash", chash)

    async def fake_embed_batch(texts):
        return [FAKE_VECTOR_BYTES for _ in texts]

    single_server = {"servers": [{"server_name": "github", "description": "GitHub"}]}
    single_tool = {"tools": [{
        "name": "search_repos",
        "description": "Search GitHub repositories",
        "inputSchema": {},
        "tags": [],
    }]}

    async with respx.mock:
        respx.get("http://fakeregistry/mcp/servers").mock(
            return_value=httpx.Response(200, json=single_server)
        )
        respx.get("http://fakeregistry/mcp/servers/github/tools").mock(
            return_value=httpx.Response(200, json=single_tool)
        )

        async with httpx.AsyncClient() as http_client:
            await run_sync_job(
                job_id="sync_test002",
                force=False,
                settings=settings,
                http_client=http_client,
                redis_client=fakeredis_client,
                embed_batch_fn=fake_embed_batch,
            )

    job = await get_sync_job_status("sync_test002", fakeredis_client)
    assert job.stats.unchanged == 1
    assert job.stats.created == 0


@pytest.mark.asyncio
async def test_start_sync_returns_job_id(fakeredis_client):
    settings = make_settings()

    async def fake_embed_batch(texts):
        return []

    async with respx.mock:
        respx.get("http://fakeregistry/mcp/servers").mock(
            return_value=httpx.Response(200, json={"servers": []})
        )

        async with httpx.AsyncClient() as http_client:
            job_id, status = await start_sync(
                force=False,
                settings=settings,
                http_client=http_client,
                redis_client=fakeredis_client,
                embed_batch_fn=fake_embed_batch,
            )

    assert job_id.startswith("sync_")
    assert status == "running"


@pytest.mark.asyncio
async def test_start_sync_returns_existing_if_locked(fakeredis_client):
    await fakeredis_client.set("sync:lock", "sync_existing123", ex=300)
    settings = make_settings()

    async with httpx.AsyncClient() as http_client:
        job_id, status = await start_sync(
            force=False,
            settings=settings,
            http_client=http_client,
            redis_client=fakeredis_client,
            embed_batch_fn=AsyncMock(return_value=[]),
        )

    assert job_id == "sync_existing123"
    assert status == "running"


@pytest.mark.asyncio
async def test_get_sync_job_status_not_found(fakeredis_client):
    result = await get_sync_job_status("nonexistent_job", fakeredis_client)
    assert result is None
