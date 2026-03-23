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
    """Covers lines 39-54: happy path with progress callbacks."""
    mock_redis = AsyncMock()
    mock_redis.hget = AsyncMock(return_value=None)

    progress_calls = []

    async def track(current, total, msg):
        progress_calls.append((current, msg))

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
            progress_callback=track,
        )

    assert result is mock_result
    assert len(progress_calls) == 2
    assert progress_calls[0][0] == 0.1
    assert progress_calls[1][0] == 0.5


@pytest.mark.asyncio
async def test_call_downstream_tool_no_progress_mock():
    """Covers lines 42-54 without progress callback branch."""
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
            progress_callback=None,
        )

    assert result is mock_result


@pytest.mark.asyncio
async def test_call_downstream_tool_exception_mock():
    """Covers lines 56-60: exception path re-raises."""
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


# ── call_tool_with_validation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_tool_not_found_mock():
    """Covers lines 82-83: tool not found raises ValueError."""
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
    """Covers lines 86-107: schema validation + downstream call."""
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
    """Covers lines 94-95: jsonschema validation error."""
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
    """Covers lines 87+ with no input_schema stored."""
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
async def test_call_tool_with_progress_mock():
    """Covers line 106: progress_callback forwarded to call_downstream_tool."""
    mock_redis = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={
        b"tool_id": b"svc:tool",
        b"name": b"tool",
        b"input_schema": b"{}",
    })
    mock_redis.hget = AsyncMock(return_value=None)

    progress_calls = []

    async def track(c, t, m):
        progress_calls.append(m)

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
            progress_callback=track,
        )

    _, kwargs = mock_call.call_args
    assert kwargs.get("progress_callback") is track


@pytest.mark.asyncio
async def test_call_tool_json_decode_error_mock():
    """Covers lines 96-97: malformed JSON schema is silently skipped."""
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
