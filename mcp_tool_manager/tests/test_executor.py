"""Tests for executor service: validation, tool call flow."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_tool_manager.services.executor import call_tool_with_validation
from mcp_tool_manager.config import Settings


def make_settings():
    settings = Settings()
    settings.registry.api_url = "http://fakeregistry"
    return settings


async def make_redis_with_tool(
    fakeredis_client,
    tool_id: str,
    schema: dict | None = None,
):
    """Helper: store a tool doc in fakeredis."""
    schema_json = json.dumps(schema or {})
    await fakeredis_client.hset(
        f"tool:{tool_id}",
        mapping={
            "tool_id": tool_id,
            "name": tool_id.split(":")[-1],
            "description": "Test tool",
            "input_schema": schema_json,
            "server_name": tool_id.split(":")[0],
        },
    )


@pytest.mark.asyncio
async def test_call_tool_invalid_tool_id(fakeredis_client):
    settings = make_settings()
    with pytest.raises(ValueError, match="Invalid tool_id"):
        await call_tool_with_validation(
            tool_id="no-colon",
            arguments={},
            redis_client=fakeredis_client,
            settings=settings,
        )


@pytest.mark.asyncio
async def test_call_tool_not_found(fakeredis_client):
    settings = make_settings()
    with pytest.raises(ValueError, match="not found"):
        await call_tool_with_validation(
            tool_id="github:nonexistent_tool",
            arguments={},
            redis_client=fakeredis_client,
            settings=settings,
        )


@pytest.mark.asyncio
async def test_call_tool_validation_failure(fakeredis_client):
    settings = make_settings()
    schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string"},
        },
    }
    await make_redis_with_tool(fakeredis_client, "github:search_repos", schema)

    with pytest.raises(ValueError, match="validation failed"):
        await call_tool_with_validation(
            tool_id="github:search_repos",
            arguments={"wrong_field": 123},  # missing required "query"
            redis_client=fakeredis_client,
            settings=settings,
        )


@pytest.mark.asyncio
async def test_call_tool_valid_args_calls_downstream(fakeredis_client):
    settings = make_settings()
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
    }
    await make_redis_with_tool(fakeredis_client, "github:search_repos", schema)

    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="result text")]

    with patch(
        "mcp_tool_manager.services.executor.call_downstream_tool",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        tool_name, result = await call_tool_with_validation(
            tool_id="github:search_repos",
            arguments={"query": "python ml"},
            redis_client=fakeredis_client,
            settings=settings,
        )

    assert tool_name == "search_repos"
    assert result is mock_result


@pytest.mark.asyncio
async def test_call_tool_empty_schema_skips_validation(fakeredis_client):
    """Empty schema = no validation, any args accepted."""
    settings = make_settings()
    await make_redis_with_tool(fakeredis_client, "github:any_tool", {})

    mock_result = MagicMock()
    mock_result.content = []

    with patch(
        "mcp_tool_manager.services.executor.call_downstream_tool",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        tool_name, result = await call_tool_with_validation(
            tool_id="github:any_tool",
            arguments={"anything": "goes"},
            redis_client=fakeredis_client,
            settings=settings,
        )
    assert tool_name == "any_tool"
