"""call_tool MCP tool registration."""
import logging
import time
import uuid

from fastmcp import Context
from openinference.semconv.trace import SpanAttributes

from mcp_tool_manager.config import get_settings
from mcp_tool_manager.dependencies import get_bedrock, get_cache_queue, get_redis
from mcp_tool_manager.main import mcp
from mcp_tool_manager.services.cache import result_to_text
from mcp_tool_manager.services.executor import call_tool_with_validation
from mcp_tool_manager.services.summariser import summarise_tool_result
from mcp_tool_manager.telemetry import get_tracer, record_counter, record_histogram

logger = logging.getLogger(__name__)


@mcp.tool()
async def call_tool(
    session_id: str,
    tool_id: str,
    arguments: dict,
    ctx: Context | None = None,
) -> dict:
    """
    Execute a discovered tool. Provide session_id to accumulate context across calls.
    tool_id format: "server_name:tool_name" (from find_tools results).
    arguments: must match the tool's input schema.
    Returns: summary of results, gaps identified, and a tool_call_id for fetching raw data.
    """
    tracer = get_tracer()
    start = time.monotonic()

    with tracer.start_as_current_span("call_tool") as span:
        span.set_attribute("tool.name", tool_id)
        span.set_attribute("tool.parameters", str(arguments))
        span.set_attribute("session.id", session_id)

        settings = get_settings()
        redis = get_redis()
        bedrock = get_bedrock()

        tool_call_id = f"{session_id}:{uuid.uuid4().hex[:8]}"
        server_name = tool_id.split(":", 1)[0]

        try:
            if ctx:
                await ctx.report_progress(0.1, 1.0, "Connecting to server...")

            # Validate + execute
            tool_name, raw_result = await call_tool_with_validation(
                tool_id=tool_id,
                arguments=arguments,
                redis_client=redis,
                settings=settings,
                ctx=ctx,
            )

            if ctx:
                await ctx.report_progress(0.5, 1.0, "Processing results...")

            result_text = result_to_text(raw_result)

            # Summarise via Bedrock
            if ctx:
                await ctx.report_progress(0.8, 1.0, "Summarising results...")

            summary_start = time.monotonic()
            summary, gaps = await summarise_tool_result(
                tool_name=tool_name,
                arguments=arguments,
                raw_result=result_text,
                bedrock_client=bedrock,
                settings=settings,
            )
            record_histogram("mcp.summary.latency_ms", (time.monotonic() - summary_start) * 1000)

            # Queue async cache write (non-blocking)
            cache_queue = get_cache_queue()
            try:
                cache_queue.put_nowait(
                    (session_id, tool_call_id, tool_id, arguments, raw_result)
                )
            except Exception as exc:
                logger.warning("Cache queue full, dropping write for %s: %s", tool_call_id, exc)

            elapsed_ms = (time.monotonic() - start) * 1000
            record_counter("mcp.tool_calls.count", attributes={"server": server_name, "tool": tool_name, "status": "success"})
            record_histogram("mcp.tool_calls.latency_ms", elapsed_ms, attributes={"server": server_name})

            span.set_attribute(SpanAttributes.OUTPUT_VALUE, summary)

            if ctx:
                await ctx.report_progress(1.0, 1.0, "Done")

            return {
                "tool_call_id": tool_call_id,
                "summary": summary,
                "gaps": gaps,
                "server": server_name,
                "tool_name": tool_name,
            }

        except Exception as exc:
            record_counter("mcp.tool_calls.errors", attributes={"server": server_name, "error_type": type(exc).__name__})
            span.record_exception(exc)
            raise
