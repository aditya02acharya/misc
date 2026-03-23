"""POST /v1/sync, GET /v1/sync/{job_id}"""
import asyncio
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mcp_tool_manager.config import get_settings
from mcp_tool_manager.dependencies import get_redis, get_httpx, get_tool_index
from mcp_tool_manager.services.registry import start_sync, get_sync_job_status
from mcp_tool_manager.services.embedding import embed_batch

router = APIRouter()
logger = logging.getLogger(__name__)


class SyncRequest(BaseModel):
    force: bool = False


@router.post("/sync")
async def trigger_sync(request: SyncRequest = SyncRequest()):
    settings = get_settings()
    redis = get_redis()
    http = get_httpx()

    async def _embed_batch(texts):
        return await embed_batch(texts, settings, http)

    job_id, status = await start_sync(
        force=request.force,
        settings=settings,
        http_client=http,
        redis_client=redis,
        embed_batch_fn=_embed_batch,
    )

    # Schedule tool_list_changed notification after sync completes
    asyncio.create_task(_notify_tool_list_changed(job_id, redis))

    return {"job_id": job_id, "status": status}


async def _notify_tool_list_changed(job_id: str, redis_client) -> None:
    """Wait for sync to complete, then notify MCP clients that tool list changed."""
    try:
        for _ in range(120):  # wait up to 120s
            job = await get_sync_job_status(job_id, redis_client)
            if job and job.status in ("completed", "failed"):
                break
            await asyncio.sleep(1)

        if job and job.status == "completed":
            from mcp_tool_manager.main import mcp
            try:
                await mcp.send_tool_list_changed()
                logger.info("Sent tool_list_changed notification after sync %s", job_id)
            except Exception as exc:
                logger.debug("Could not send tool_list_changed: %s", exc)
    except Exception as exc:
        logger.debug("tool_list_changed notification task failed: %s", exc)


@router.get("/sync/{job_id}")
async def get_sync_status(job_id: str):
    redis = get_redis()
    job = await get_sync_job_status(job_id, redis)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Sync job {job_id!r} not found")
    return job.model_dump()
