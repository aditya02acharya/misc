"""fetch_cached_data MCP tool registration."""
import logging

from fastmcp import Context

from mcp_tool_manager.config import get_settings
from mcp_tool_manager.dependencies import get_cache_index, get_httpx, get_redis
from mcp_tool_manager.main import mcp
from mcp_tool_manager.services.fetch import (
    fetch_all_session,
    fetch_by_query,
    fetch_by_tool_call_id,
)
from mcp_tool_manager.telemetry import record_counter

logger = logging.getLogger(__name__)


@mcp.tool()
async def fetch_cached_data(
    session_id: str,
    tool_call_id: str | None = None,
    query: str | None = None,
    top_k: int = 5,
    ctx: Context | None = None,
) -> dict:
    """
    Retrieve cached context from previous tool calls in this session.
    - No filters: returns all cached chunks for the session (most recent first).
    - tool_call_id: returns chunks from a specific call.
    - query: semantic search across all session data, returns top_k matches.
    """
    if ctx:
        await ctx.report_progress(0.1, 1.0, "Fetching cached data...")

    settings = get_settings()
    redis = get_redis()

    if tool_call_id:
        result = await fetch_by_tool_call_id(session_id, tool_call_id, redis)
    elif query:
        result = await fetch_by_query(
            session_id=session_id,
            query=query,
            top_k=top_k,
            settings=settings,
            cache_index=get_cache_index(),
            http_client=get_httpx(),
            redis_client=redis,
        )
    else:
        result = await fetch_all_session(session_id, top_k, redis)

    record_counter("mcp.cache.reads", value=len(result.chunks))

    if ctx:
        await ctx.report_progress(1.0, 1.0, f"Retrieved {result.total_chunks} chunks")

    return {
        "chunks": [c.model_dump(exclude={"embedding"}) for c in result.chunks],
        "total_chunks": result.total_chunks,
    }
