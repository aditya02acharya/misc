"""find_tools MCP tool registration."""
import logging
import time

from fastmcp import Context
from openinference.semconv.trace import SpanAttributes

from mcp_tool_manager.config import get_settings
from mcp_tool_manager.dependencies import get_httpx, get_redis, get_tool_index
from mcp_tool_manager.main import mcp
from mcp_tool_manager.services.search import hybrid_search
from mcp_tool_manager.telemetry import get_tracer, record_counter, record_histogram

logger = logging.getLogger(__name__)


@mcp.tool()
async def find_tools(
    query: str,
    top_k: int = 10,
    server_filter: str | None = None,
    ctx: Context | None = None,
) -> list[dict]:
    """
    Discover available tools. Describe WHAT you need to accomplish, not HOW.
    Use @hint to narrow results — hint matches against server names, tool names,
    and tags (e.g., "@github", "@search_repos", "@api", "@database").
    Multiple @hints supported: "@github @search find code repos".
    Returns ranked tools with name, description, server, and input schema summary.
    """
    tracer = get_tracer()
    start = time.monotonic()

    with tracer.start_as_current_span("find_tools") as span:
        span.set_attribute(SpanAttributes.INPUT_VALUE, query)

        if ctx:
            await ctx.report_progress(0.1, 1.0, "Searching tools...")

        settings = get_settings()
        results = await hybrid_search(
            query=query,
            top_k=top_k,
            settings=settings,
            tool_index=get_tool_index(),
            http_client=get_httpx(),
            redis_client=get_redis(),
            server_filter=server_filter,
        )

        output = [r.model_dump() for r in results]
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(output))
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.documents", str(len(results)))

        elapsed_ms = (time.monotonic() - start) * 1000
        record_counter("mcp.search.count")
        record_histogram("mcp.search.latency_ms", elapsed_ms)

        if ctx:
            await ctx.report_progress(1.0, 1.0, f"Found {len(results)} tools")

        return output
