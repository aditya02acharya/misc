"""Extended executor tests: endpoint resolution, downstream call, session cleanup."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_tool_manager.services.executor import (
    _get_server_endpoint,
    call_downstream_tool,
    close_all_sessions,
)
from mcp_tool_manager.config import Settings


def make_settings():
    s = Settings()
    s.registry.api_url = "http://fakeregistry"
    return s


# ── _get_server_endpoint ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_server_endpoint_from_redis(fakeredis_client):
    await fakeredis_client.hset("server:github", "endpoint", "http://github-mcp:8080")
    settings = make_settings()

    endpoint = await _get_server_endpoint("github", fakeredis_client, settings)
    assert endpoint == "http://github-mcp:8080"


@pytest.mark.asyncio
async def test_get_server_endpoint_fallback(fakeredis_client):
    settings = make_settings()

    endpoint = await _get_server_endpoint("unknown_server", fakeredis_client, settings)
    assert endpoint == "http://fakeregistry/mcp/servers/unknown_server"


# ── call_downstream_tool ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_downstream_tool_with_ctx(fakeredis_client):
    import mcp_tool_manager.services.executor as ex
    ex._session_pool.clear()

    settings = make_settings()

    mock_result = MagicMock()
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    mock_session.initialize = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_transport = AsyncMock()
    mock_transport.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock(), None))
    mock_transport.__aexit__ = AsyncMock(return_value=False)

    mock_ctx = MagicMock()

    with patch("mcp.client.streamable_http.streamablehttp_client", return_value=mock_transport), \
         patch("mcp.ClientSession", return_value=mock_session):

        result = await call_downstream_tool(
            server_name="github",
            tool_name="search_repos",
            arguments={"query": "python"},
            redis_client=fakeredis_client,
            settings=settings,
            ctx=mock_ctx,
        )

    assert result is mock_result
    ex._session_pool.clear()


@pytest.mark.asyncio
async def test_call_downstream_tool_raises_on_failure(fakeredis_client):
    import mcp_tool_manager.services.executor as ex
    ex._session_pool.clear()

    settings = make_settings()

    mock_transport = AsyncMock()
    mock_transport.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
    mock_transport.__aexit__ = AsyncMock(return_value=False)

    with patch("mcp.client.streamable_http.streamablehttp_client", return_value=mock_transport):
        with pytest.raises(ConnectionError, match="refused"):
            await call_downstream_tool(
                server_name="broken",
                tool_name="any_tool",
                arguments={},
                redis_client=fakeredis_client,
                settings=settings,
            )
    ex._session_pool.clear()


@pytest.mark.asyncio
async def test_call_tool_with_validation_ctx_forwarded(fakeredis_client):
    """ctx is passed through to call_downstream_tool."""
    import json as _json
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    await fakeredis_client.hset(
        "tool:svc:tool",
        mapping={"tool_id": "svc:tool", "input_schema": _json.dumps(schema), "name": "tool"},
    )

    mock_ctx = MagicMock()

    with patch(
        "mcp_tool_manager.services.executor.call_downstream_tool",
        new_callable=AsyncMock,
        return_value=MagicMock(content=[]),
    ) as mock_call:
        from mcp_tool_manager.services.executor import call_tool_with_validation
        await call_tool_with_validation(
            tool_id="svc:tool",
            arguments={"q": "hello"},
            redis_client=fakeredis_client,
            settings=Settings(),
            ctx=mock_ctx,
        )

    # ctx was forwarded
    _, kwargs = mock_call.call_args
    assert kwargs.get("ctx") is mock_ctx


@pytest.mark.asyncio
async def test_call_tool_with_validation_json_decode_error(fakeredis_client):
    """Malformed schema JSON is silently skipped, call proceeds."""
    await fakeredis_client.hset(
        "tool:svc:bad_schema_tool",
        mapping={"tool_id": "svc:bad_schema_tool", "input_schema": "NOT_JSON", "name": "bad_schema_tool"},
    )

    with patch(
        "mcp_tool_manager.services.executor.call_downstream_tool",
        new_callable=AsyncMock,
        return_value=MagicMock(content=[]),
    ):
        from mcp_tool_manager.services.executor import call_tool_with_validation
        tool_name, result = await call_tool_with_validation(
            tool_id="svc:bad_schema_tool",
            arguments={"anything": "goes"},
            redis_client=fakeredis_client,
            settings=Settings(),
        )
    assert tool_name == "bad_schema_tool"


# ── close_all_sessions ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_all_sessions_empty():
    import mcp_tool_manager.services.executor as ex
    ex._session_pool = {}
    await close_all_sessions()  # Should not raise


@pytest.mark.asyncio
async def test_close_all_sessions_with_entries():
    import mcp_tool_manager.services.executor as ex
    mock_session = AsyncMock()
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_transport = AsyncMock()
    mock_transport.__aexit__ = AsyncMock(return_value=False)
    ex._session_pool = {"github": (mock_session, mock_transport)}

    await close_all_sessions()

    mock_session.__aexit__.assert_called_once()
    assert ex._session_pool == {}


@pytest.mark.asyncio
async def test_close_all_sessions_ignores_close_errors():
    import mcp_tool_manager.services.executor as ex
    mock_session = AsyncMock()
    mock_session.__aexit__ = AsyncMock(side_effect=Exception("close failed"))
    mock_transport = AsyncMock()
    mock_transport.__aexit__ = AsyncMock(side_effect=Exception("close failed"))
    ex._session_pool = {"broken": (mock_session, mock_transport)}

    await close_all_sessions()  # Should not raise
    assert ex._session_pool == {}
