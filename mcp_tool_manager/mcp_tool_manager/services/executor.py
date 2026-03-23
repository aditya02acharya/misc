"""MCP ClientSession pool: create and reuse sessions per downstream server."""
import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Pool: server_name -> (session, transport context manager)
_session_pool: dict[str, Any] = {}
_session_lock = asyncio.Lock()


async def _get_server_endpoint(server_name: str, redis_client: Any, settings: Any) -> str:
    """Resolve server endpoint from Redis tool doc or settings."""
    # Try to get from a stored server metadata key
    endpoint = await redis_client.hget(f"server:{server_name}", "endpoint")
    if endpoint:
        return endpoint.decode() if isinstance(endpoint, bytes) else endpoint

    # Fall back to registry URL pattern
    return f"{settings.registry.api_url}/mcp/servers/{server_name}"


async def call_downstream_tool(
    server_name: str,
    tool_name: str,
    arguments: dict,
    redis_client: Any,
    settings: Any,
    progress_callback=None,
) -> Any:
    """
    Call a tool on a downstream MCP server.
    Returns the raw result from the MCP server.
    """
    endpoint = await _get_server_endpoint(server_name, redis_client, settings)

    if progress_callback:
        await progress_callback(0.1, 1.0, "Connecting to server...")

    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(endpoint) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                if progress_callback:
                    await progress_callback(0.5, 1.0, "Executing tool...")

                result = await session.call_tool(tool_name, arguments)
                return result

    except Exception as exc:
        logger.error(
            "Failed to call %s:%s — %s", server_name, tool_name, exc
        )
        raise


async def call_tool_with_validation(
    tool_id: str,
    arguments: dict,
    redis_client: Any,
    settings: Any,
    progress_callback=None,
) -> tuple[str, Any]:
    """
    Full call flow: validate → connect → execute.
    Returns (tool_name, raw_result).
    """
    parts = tool_id.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid tool_id format: {tool_id!r}. Expected 'server_name:tool_name'")

    server_name, tool_name = parts

    # Fetch tool doc from Redis for schema validation
    tool_data = await redis_client.hgetall(f"tool:{tool_id}")
    if not tool_data:
        raise ValueError(f"Tool {tool_id!r} not found in index. Run /v1/sync first.")

    # Validate arguments against input_schema
    schema_raw = tool_data.get(b"input_schema") or tool_data.get("input_schema")
    if schema_raw:
        schema_str = schema_raw.decode() if isinstance(schema_raw, bytes) else schema_raw
        try:
            import jsonschema
            schema = json.loads(schema_str)
            if schema:
                jsonschema.validate(arguments, schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(f"Argument validation failed: {exc.message}") from exc
        except json.JSONDecodeError:
            pass  # Malformed schema stored — skip validation

    raw_result = await call_downstream_tool(
        server_name=server_name,
        tool_name=tool_name,
        arguments=arguments,
        redis_client=redis_client,
        settings=settings,
        progress_callback=progress_callback,
    )
    return tool_name, raw_result


async def close_all_sessions() -> None:
    """Close all pooled MCP sessions on shutdown."""
    global _session_pool
    for name, session in list(_session_pool.items()):
        try:
            await session.close()
        except Exception:
            pass
    _session_pool.clear()
    logger.info("All MCP sessions closed")
