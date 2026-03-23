"""Coverage tests for executor.py using AsyncMock (fixes async coverage tracking)."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_tool_manager.services.executor import (
    _get_server_endpoint,
    call_downstream_tool,
    call_tool_with_validation,
)
from mcp_tool_manager.config import Settings


def make_settings():
    s = Settings()
    s.registry.api_url = "http://fakeregistry"
    return s


# ── _get_server_endpoint ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_server_endpoint_redis_hit_mock():
    """Covers lines 18-19: endpoint found in Redis."""
    mock_redis = AsyncMock()
    mock_redis.hget = AsyncMock(return_value=b"http://github-mcp:8080")

    result = await _get_server_endpoint("github", mock_redis, make_settings())
    assert result == "http://github-mcp:8080"


@pytest.mark.asyncio
async def test_get_server_endpoint_redis_miss_mock():
    """Covers line 22: fallback when no Redis endpoint."""
    mock_redis = AsyncMock()
    mock_redis.hget = AsyncMock(return_value=None)

    result = await _get_server_endpoint("unknown", mock_redis, make_settings())
    assert result == "http://fakeregistry/mcp/servers/unknown"


@pytest.mark.asyncio
async def test_get_server_endpoint_string_endpoint_mock():
    """Covers line 19: non-bytes endpoint (already decoded)."""
    mock_redis = AsyncMock()
    mock_redis.hget = AsyncMock(return_value="http://string-endpoint:9090")

    result = await _get_server_endpoint("svc", mock_redis, make_settings())
    assert result == "http://string-endpoint:9090"


# ── call_downstream_tool ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_downstream_tool_success_mock():
    """Covers happy path with pooled session."""
    import mcp_tool_manager.services.executor as ex
    # Clear pool before test
    ex._session_pool.clear()

    mock_redis = AsyncMock()
    mock_redis.hget = AsyncMock(return_value=None)

    mock_result = MagicMock()
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    mock_session.initialize = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_transport = AsyncMock()
    mock_transport.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock(), None))
    mock_transport.__aexit__ = AsyncMock(return_value=False)

    with patch("mcp.client.streamable_http.streamablehttp_client", return_value=mock_transport), \
         patch("mcp.ClientSession", return_value=mock_session):
        result = await call_downstream_tool(
            server_name="github",
            tool_name="search_repos",
            arguments={"query": "python"},
            redis_client=mock_redis,
            settings=make_settings(),
        )

    assert result is mock_result
    # Clean up pool
    ex._session_pool.clear()


@pytest.mark.asyncio
async def test_call_downstream_tool_no_ctx_mock():
    """Covers call without ctx parameter."""
    import mcp_tool_manager.services.executor as ex
    ex._session_pool.clear()

    mock_redis = AsyncMock()
    mock_redis.hget = AsyncMock(return_value=b"http://server:8080")

    mock_result = MagicMock()
    mock_session = AsyncMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    mock_session.initialize = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_transport = AsyncMock()
    mock_transport.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock(), None))
    mock_transport.__aexit__ = AsyncMock(return_value=False)

    with patch("mcp.client.streamable_http.streamablehttp_client", return_value=mock_transport), \
         patch("mcp.ClientSession", return_value=mock_session):
        result = await call_downstream_tool(
            server_name="github",
            tool_name="search_repos",
            arguments={"query": "python"},
            redis_client=mock_redis,
            settings=make_settings(),
            ctx=None,
        )

    assert result is mock_result
    ex._session_pool.clear()


@pytest.mark.asyncio
async def test_call_downstream_tool_exception_mock():
    """Covers exception path re-raises."""
    import mcp_tool_manager.services.executor as ex
    ex._session_pool.clear()

    mock_redis = AsyncMock()
    mock_redis.hget = AsyncMock(return_value=None)

    mock_transport = AsyncMock()
    mock_transport.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
    mock_transport.__aexit__ = AsyncMock(return_value=False)

    with patch("mcp.client.streamable_http.streamablehttp_client", return_value=mock_transport):
        with pytest.raises(ConnectionError, match="refused"):
            await call_downstream_tool(
                server_name="broken",
                tool_name="any",
                arguments={},
                redis_client=mock_redis,
                settings=make_settings(),
            )
    ex._session_pool.clear()


# ── call_tool_with_validation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_tool_not_found_mock():
    """Covers tool not found raises ValueError."""
    mock_redis = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={})

    with pytest.raises(ValueError, match="not found"):
        await call_tool_with_validation(
            tool_id="svc:missing",
            arguments={},
            redis_client=mock_redis,
            settings=make_settings(),
        )


@pytest.mark.asyncio
async def test_call_tool_valid_with_schema_mock():
    """Covers schema validation + downstream call."""
    mock_redis = AsyncMock()
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    mock_redis.hgetall = AsyncMock(return_value={
        b"tool_id": b"svc:tool",
        b"name": b"tool",
        b"input_schema": json.dumps(schema).encode(),
    })
    mock_redis.hget = AsyncMock(return_value=None)

    mock_result = MagicMock()
    mock_result.content = []

    with patch(
        "mcp_tool_manager.services.executor.call_downstream_tool",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        tool_name, result = await call_tool_with_validation(
            tool_id="svc:tool",
            arguments={"q": "hello"},
            redis_client=mock_redis,
            settings=make_settings(),
        )

    assert tool_name == "tool"
    assert result is mock_result


@pytest.mark.asyncio
async def test_call_tool_schema_validation_error_mock():
    """Covers jsonschema validation error."""
    mock_redis = AsyncMock()
    schema = {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string"}},
    }
    mock_redis.hgetall = AsyncMock(return_value={
        b"tool_id": b"svc:tool",
        b"name": b"tool",
        b"input_schema": json.dumps(schema).encode(),
    })

    with pytest.raises(ValueError, match="validation failed"):
        await call_tool_with_validation(
            tool_id="svc:tool",
            arguments={"wrong_field": 123},
            redis_client=mock_redis,
            settings=make_settings(),
        )


@pytest.mark.asyncio
async def test_call_tool_no_schema_mock():
    """Covers no input_schema stored."""
    mock_redis = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={
        b"tool_id": b"svc:tool",
        b"name": b"tool",
    })
    mock_redis.hget = AsyncMock(return_value=None)

    mock_result = MagicMock()
    mock_result.content = []

    with patch(
        "mcp_tool_manager.services.executor.call_downstream_tool",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        tool_name, result = await call_tool_with_validation(
            tool_id="svc:tool",
            arguments={"anything": "goes"},
            redis_client=mock_redis,
            settings=make_settings(),
        )

    assert tool_name == "tool"


@pytest.mark.asyncio
async def test_call_tool_with_ctx_mock():
    """Covers ctx forwarded to call_downstream_tool."""
    mock_redis = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={
        b"tool_id": b"svc:tool",
        b"name": b"tool",
        b"input_schema": b"{}",
    })
    mock_redis.hget = AsyncMock(return_value=None)

    mock_ctx = MagicMock()

    mock_result = MagicMock()
    mock_result.content = []

    with patch(
        "mcp_tool_manager.services.executor.call_downstream_tool",
        new_callable=AsyncMock,
        return_value=mock_result,
    ) as mock_call:
        await call_tool_with_validation(
            tool_id="svc:tool",
            arguments={},
            redis_client=mock_redis,
            settings=make_settings(),
            ctx=mock_ctx,
        )

    _, kwargs = mock_call.call_args
    assert kwargs.get("ctx") is mock_ctx


@pytest.mark.asyncio
async def test_call_tool_json_decode_error_mock():
    """Covers malformed JSON schema is silently skipped."""
    mock_redis = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={
        b"tool_id": b"svc:tool",
        b"name": b"tool",
        b"input_schema": b"NOT_VALID_JSON",
    })
    mock_redis.hget = AsyncMock(return_value=None)

    mock_result = MagicMock()
    mock_result.content = []

    with patch(
        "mcp_tool_manager.services.executor.call_downstream_tool",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        tool_name, result = await call_tool_with_validation(
            tool_id="svc:tool",
            arguments={"q": "test"},
            redis_client=mock_redis,
            settings=make_settings(),
        )
    assert tool_name == "tool"
