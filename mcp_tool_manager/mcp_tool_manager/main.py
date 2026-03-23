"""Entrypoint: mount FastMCP + FastAPI, lifespan."""
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastmcp import FastMCP

from mcp_tool_manager.config import get_settings
from mcp_tool_manager.dependencies import (
    close_dependencies,
    init_dependencies,
    init_redis_indexes,
)
from mcp_tool_manager.telemetry import setup_telemetry

logger = logging.getLogger(__name__)

# MCP server instance (tools registered here)
mcp = FastMCP("mcp-tool-manager")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    # Telemetry
    setup_telemetry(settings)

    # Dependencies (Redis, HTTPX, Bedrock, Queue)
    await init_dependencies(settings)

    # Redis search indexes
    await init_redis_indexes(settings)

    # Import tools so they register with mcp
    import mcp_tool_manager.tools.find  # noqa: F401
    import mcp_tool_manager.tools.call  # noqa: F401
    import mcp_tool_manager.tools.fetch  # noqa: F401

    # Start background cache writer
    from mcp_tool_manager.services.cache import cache_writer_task
    from mcp_tool_manager.dependencies import get_cache_queue

    writer_task = asyncio.create_task(cache_writer_task(get_cache_queue(), settings))

    logger.info("MCP Tool Manager started")
    yield

    # Shutdown
    writer_task.cancel()
    try:
        await writer_task
    except asyncio.CancelledError:
        pass

    # Close MCP executor sessions
    try:
        from mcp_tool_manager.services.executor import close_all_sessions
        await close_all_sessions()
    except Exception:
        pass

    await close_dependencies()
    logger.info("MCP Tool Manager stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    # REST API routers
    from mcp_tool_manager.api.sync import router as sync_router
    from mcp_tool_manager.api.health import router as health_router
    from mcp_tool_manager.api.metrics import router as metrics_router

    app.include_router(sync_router, prefix="/v1")
    app.include_router(health_router, prefix="/v1")
    app.include_router(metrics_router, prefix="/v1")

    # Mount FastMCP as ASGI app
    app.mount("/mcp", mcp.http_app())

    return app


def run():
    settings = get_settings()
    uvicorn.run(
        "mcp_tool_manager.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
