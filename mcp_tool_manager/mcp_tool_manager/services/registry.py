"""LiteLLM registry client + sync job logic."""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone

import httpx

from mcp_tool_manager.models import SyncJobStatus, SyncResult, ToolDoc

logger = logging.getLogger(__name__)


def _content_hash(server_name: str, tool_name: str, description: str, schema: dict) -> str:
    canonical = {
        "server": server_name,
        "name": tool_name,
        "description": description,
        "schema": schema,
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:16]


def _compose_embedding_text(tool: ToolDoc) -> str:
    """Single combined embedding text per tool (Section 5.1 pattern)."""
    params = []
    props = tool.input_schema.get("properties", {})
    for i, (pname, pdef) in enumerate(props.items()):
        if i >= 5:
            break
        pdesc = pdef.get("description", "")
        params.append(f"{pname}: {pdesc}" if pdesc else pname)

    parts = [
        f"Tool: {tool.name}",
        f"Function: {tool.name}",
        f"Description: {tool.description}",
        f"Server: {tool.server_name} — {tool.server_description}",
        f"Capabilities: {', '.join(tool.tags)}",
        f"Parameters: {'; '.join(params)}",
    ]
    return "\n".join(parts)


async def fetch_servers(settings, http_client: httpx.AsyncClient) -> list[dict]:
    """Fetch all MCP servers from LiteLLM registry."""
    headers = {}
    if settings.registry.api_key:
        headers["Authorization"] = f"Bearer {settings.registry.api_key}"

    response = await http_client.get(
        f"{settings.registry.api_url}/mcp/servers",
        headers=headers,
        timeout=settings.registry.sync_timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("servers", data) if isinstance(data, dict) else data


async def fetch_tools_for_server(
    server: dict, settings, http_client: httpx.AsyncClient
) -> list[ToolDoc]:
    """Fetch tools from a single MCP server via LiteLLM."""
    server_name = server.get("server_name") or server.get("name", "unknown")
    server_description = server.get("description", "")
    server_url = server.get("url") or server.get("endpoint", "")

    headers = {}
    if settings.registry.api_key:
        headers["Authorization"] = f"Bearer {settings.registry.api_key}"

    try:
        response = await http_client.get(
            f"{settings.registry.api_url}/mcp/servers/{server_name}/tools",
            headers=headers,
            timeout=settings.registry.sync_timeout,
        )
        response.raise_for_status()
        data = response.json()
        tools_data = data.get("tools", data) if isinstance(data, dict) else data
    except Exception as exc:
        logger.warning("Failed to fetch tools for server %s: %s", server_name, exc)
        return []

    tools = []
    for t in tools_data:
        tool_name = t.get("name", "")
        description = t.get("description", "")
        input_schema = t.get("inputSchema") or t.get("input_schema") or {}
        tags_raw = t.get("tags", [])
        tags = tags_raw if isinstance(tags_raw, list) else [tags_raw]

        doc = ToolDoc(
            tool_id=f"{server_name}:{tool_name}",
            name=tool_name,
            description=description,
            server_name=server_name,
            server_description=server_description,
            tags=tags,
            input_schema=input_schema,
            content_hash=_content_hash(server_name, tool_name, description, input_schema),
        )
        doc.searchable_text = _compose_embedding_text(doc)
        tools.append(doc)

    return tools


async def run_sync_job(
    job_id: str,
    force: bool,
    settings,
    http_client: httpx.AsyncClient,
    redis_client,
    embed_batch_fn,
) -> None:
    """Background sync task. Stores job status in Redis."""
    start = time.monotonic()
    job_key = f"sync:job:{job_id}"

    async def update_status(status: SyncJobStatus):
        await redis_client.setex(job_key, 3600, status.model_dump_json())

    job = SyncJobStatus(job_id=job_id, status="running")
    await update_status(job)

    try:
        # Fetch all servers and tools
        servers = await fetch_servers(settings, http_client)
        all_tools: list[ToolDoc] = []
        for server in servers:
            tools = await fetch_tools_for_server(server, settings, http_client)
            all_tools.extend(tools)

        stats = SyncResult(fetched=len(all_tools))

        # Fetch existing content hashes from Redis (pipeline)
        existing_hashes: dict[str, str] = {}
        if all_tools:
            pipe = redis_client.pipeline()
            for tool in all_tools:
                pipe.hget(f"tool:{tool.tool_id}", "content_hash")
            hash_results = await pipe.execute()
            for tool, h in zip(all_tools, hash_results):
                if h:
                    existing_hashes[tool.tool_id] = h.decode() if isinstance(h, bytes) else h

        # Diff: new / updated / unchanged
        to_embed: list[ToolDoc] = []
        unchanged: list[ToolDoc] = []
        for tool in all_tools:
            existing_hash = existing_hashes.get(tool.tool_id)
            if existing_hash == tool.content_hash and not force:
                unchanged.append(tool)
            else:
                to_embed.append(tool)

        stats.unchanged = len(unchanged)

        # Batch embed only new+updated tools
        embed_start = time.monotonic()
        if to_embed:
            texts = [t.searchable_text for t in to_embed]
            embeddings = await embed_batch_fn(texts)
            for tool, emb in zip(to_embed, embeddings):
                tool.embedding = emb
        stats.embedding_time_ms = (time.monotonic() - embed_start) * 1000

        # Store in Redis
        pipe = redis_client.pipeline()
        for tool in to_embed:
            key = f"tool:{tool.tool_id}"
            existing = existing_hashes.get(tool.tool_id)
            if existing is None:
                stats.created += 1
            else:
                stats.updated += 1

            data = {
                "tool_id": tool.tool_id,
                "name": tool.name,
                "name_tag": tool.name.lower(),
                "description": tool.description,
                "searchable_text": tool.searchable_text,
                "server_name": tool.server_name,
                "tags": ",".join(tool.tags),
                "embedding": tool.embedding,
                "input_schema": json.dumps(tool.input_schema),
                "content_hash": tool.content_hash,
            }
            pipe.hset(key, mapping=data)

        await pipe.execute()

        # Delete removed tools
        current_ids = {f"tool:{t.tool_id}" for t in all_tools}
        cursor = 0
        deleted_count = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match="tool:*", count=100)
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                if key_str not in current_ids:
                    await redis_client.delete(key_str)
                    deleted_count += 1
            if cursor == 0:
                break
        stats.deleted = deleted_count
        stats.total_time_ms = (time.monotonic() - start) * 1000

        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.stats = stats
        await update_status(job)

        # Record last successful sync timestamp
        await redis_client.set("sync:last_completed", datetime.now(timezone.utc).isoformat())

        # Update tools.total gauge
        from mcp_tool_manager.telemetry import record_gauge, record_counter
        record_gauge("mcp.tools.total", float(len(all_tools)))
        record_counter("mcp.sync.runs")

        logger.info("Sync completed: %s", stats.model_dump())

    except Exception as exc:
        logger.exception("Sync job %s failed", job_id)
        job.status = "failed"
        job.error = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        await update_status(job)
    finally:
        # Release lock
        await redis_client.delete("sync:lock")


async def start_sync(
    force: bool,
    settings,
    http_client: httpx.AsyncClient,
    redis_client,
    embed_batch_fn,
) -> tuple[str, str]:
    """
    Start a sync job or return existing running job.
    Returns (job_id, status).
    """
    # Check for existing lock
    existing_lock = await redis_client.get("sync:lock")
    if existing_lock and not force:
        job_id = existing_lock.decode() if isinstance(existing_lock, bytes) else existing_lock
        return job_id, "running"

    # Acquire lock
    job_id = f"sync_{uuid.uuid4().hex[:8]}"
    acquired = await redis_client.set("sync:lock", job_id, nx=True, ex=300)
    if not acquired:
        # Race condition: someone else got the lock
        existing_lock = await redis_client.get("sync:lock")
        job_id = existing_lock.decode() if isinstance(existing_lock, bytes) else existing_lock
        return job_id, "running"

    # Launch background task
    asyncio.create_task(
        run_sync_job(job_id, force, settings, http_client, redis_client, embed_batch_fn)
    )
    return job_id, "running"


async def get_sync_job_status(job_id: str, redis_client) -> SyncJobStatus | None:
    """Fetch sync job status from Redis."""
    data = await redis_client.get(f"sync:job:{job_id}")
    if data is None:
        return None
    return SyncJobStatus.model_validate_json(data)
