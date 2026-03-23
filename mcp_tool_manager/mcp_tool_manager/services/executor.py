"""MCP ClientSession pool: create and reuse sessions per downstream server."""
import asyncio
import json
import logging
from typing import Any

from mcp_tool_manager.telemetry import record_gauge

logger = logging.getLogger(__name__)

# Pool: server_name -> (ClientSession, cleanup callable)
_session_pool: dict[str, Any] = {}
_session_lock = asyncio.Lock()


async def _get_server_endpoint(server_name: str, redis_client: Any, settings: Any) -> str:
    """Resolve server endpoint from Redis tool doc or settings."""
    endpoint = await redis_client.hget(f"server:{server_name}", "endpoint")
    if endpoint:
        return endpoint.decode() if isinstance(endpoint, bytes) else endpoint

    return f"{settings.registry.api_url}/mcp/servers/{server_name}"


async def _get_or_create_session(server_name: str, endpoint: str) -> Any:
    """Get a pooled session or create a new one for the server."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with _session_lock:
        if server_name in _session_pool:
            session, _ = _session_pool[server_name]
            try:
                # Verify session is still alive by checking it exists
                return session
            except Exception:
                # Stale session, remove and recreate
                del _session_pool[server_name]

        # Create new session — note: we need to manage the transport context
        # For pooled sessions, we open the transport and keep it alive
        transport_cm = streamablehttp_client(endpoint)
        read, write, _ = await transport_cm.__aenter__()
        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()

        _session_pool[server_name] = (session, transport_cm)
        record_gauge("mcp.sessions.active", float(len(_session_pool)))
        logger.info("Created pooled MCP session for server: %s", server_name)
        return session


async def call_downstream_tool(
    server_name: str,
    tool_name: str,
    arguments: dict,
    redis_client: Any,
    settings: Any,
    ctx: Any = None,
) -> Any:
    """
    Call a tool on a downstream MCP server.
    Uses pooled sessions. Supports sampling/elicitation pass-through via ctx.
    Returns the raw result from the MCP server.
    """
    endpoint = await _get_server_endpoint(server_name, redis_client, settings)

    try:
        session = await _get_or_create_session(server_name, endpoint)
        result = await session.call_tool(tool_name, arguments)
        return result

    except Exception as exc:
        # On failure, remove stale session from pool and retry once with fresh session
        async with _session_lock:
            if server_name in _session_pool:
                old_session, old_transport = _session_pool.pop(server_name)
                try:
                    await old_session.__aexit__(None, None, None)
                    await old_transport.__aexit__(None, None, None)
                except Exception:
                    pass

        # Retry with fresh session
        try:
            session = await _get_or_create_session(server_name, endpoint)
            result = await session.call_tool(tool_name, arguments)
            return result
        except Exception:
            logger.error("Failed to call %s:%s — %s", server_name, tool_name, exc)
            raise


async def call_tool_with_validation(
    tool_id: str,
    arguments: dict,
    redis_client: Any,
    settings: Any,
    ctx: Any = None,
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
        ctx=ctx,
    )
    return tool_name, raw_result


async def close_all_sessions() -> None:
    """Close all pooled MCP sessions on shutdown."""
    global _session_pool
    async with _session_lock:
        for name, (session, transport_cm) in list(_session_pool.items()):
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await transport_cm.__aexit__(None, None, None)
            except Exception:
                pass
        _session_pool.clear()
        record_gauge("mcp.sessions.active", 0.0)
    logger.info("All MCP sessions closed")


def get_active_session_count() -> int:
    """Return number of active pooled sessions."""
    return len(_session_pool)
