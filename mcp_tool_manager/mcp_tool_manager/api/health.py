"""GET /v1/health"""
import logging
from datetime import datetime

from fastapi import APIRouter

from mcp_tool_manager.config import get_settings
from mcp_tool_manager.dependencies import get_redis, get_httpx

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    settings = get_settings()
    redis_status = "ok"
    vllm_status = "ok"
    registry_status = "ok"
    tool_count = 0
    last_sync = None

    # Redis check
    try:
        redis = get_redis()
        await redis.ping()
        # Count tools
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match="tool:*", count=100)
            tool_count += len(keys)
            if cursor == 0:
                break
        # Last sync info
        last_sync_raw = await redis.get("sync:last_completed")
        if last_sync_raw:
            last_sync = last_sync_raw.decode()
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        redis_status = f"error: {exc}"

    # vLLM check (HEAD request)
    try:
        http = get_httpx()
        resp = await http.get(settings.embedding.api_url.replace("/embeddings", "/models"))
        vllm_status = "ok" if resp.status_code < 500 else f"error: {resp.status_code}"
    except Exception as exc:
        vllm_status = f"error: {exc}"

    # Registry check
    try:
        http = get_httpx()
        resp = await http.get(f"{settings.registry.api_url}/health", timeout=5.0)
        registry_status = "ok" if resp.status_code < 500 else f"error: {resp.status_code}"
    except Exception as exc:
        registry_status = f"error: {exc}"

    overall = (
        "healthy"
        if all(s == "ok" for s in [redis_status, vllm_status, registry_status])
        else "degraded"
    )

    return {
        "status": overall,
        "redis": redis_status,
        "vllm": vllm_status,
        "registry": registry_status,
        "tool_count": tool_count,
        "last_sync": last_sync,
    }
