"""Tests for hybrid search service."""
import struct
import pytest
import respx
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_tool_manager.services.search import (
    parse_hints,
    _build_hint_filter,
    _apply_substring_boost,
    _input_schema_summary,
    hybrid_search,
)
from mcp_tool_manager.config import Settings


FAKE_VECTOR = b"\x00" * (1024 * 4)


def make_settings(**overrides):
    import os
    env_overrides = {
        "EMBEDDING__API_URL": "http://fakeembed/v1/embeddings",
    }
    env_overrides.update(overrides)
    old = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v
    s = Settings()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return s


def test_parse_hints_none():
    hints, clean = parse_hints("search github repositories")
    assert hints == []
    assert clean == "search github repositories"


def test_parse_hints_single():
    hints, clean = parse_hints("@github search repositories")
    assert hints == ["github"]
    assert clean == "search repositories"


def test_parse_hints_multiple():
    hints, clean = parse_hints("@github @api find tools")
    assert "github" in hints
    assert "api" in hints
    assert clean == "find tools"


def test_parse_hints_only_hints():
    hints, clean = parse_hints("@github @search")
    assert len(hints) == 2
    assert clean == ""


def test_parse_hints_end_of_query():
    hints, clean = parse_hints("search repos @github")
    assert hints == ["github"]
    assert "search repos" in clean


def test_apply_substring_boost_no_hints():
    items = [{"name": "search_repos", "server_name": "github", "tags": "search", "score": 0.8}]
    result = _apply_substring_boost(items, [])
    # When no hints provided, items returned unchanged (no _boosted_score key)
    assert result == items


def test_apply_substring_boost_matching_name():
    items = [
        {"name": "search_repos", "server_name": "github", "tags": "api", "vector_distance": 0.5},
        {"name": "create_issue", "server_name": "github", "tags": "write", "vector_distance": 0.5},
    ]
    result = _apply_substring_boost(items, ["search"])
    # search_repos should be boosted (contains "search")
    search_item = next(r for r in result if r["name"] == "search_repos")
    create_item = next(r for r in result if r["name"] == "create_issue")
    assert search_item["_boosted_score"] > create_item["_boosted_score"]


def test_apply_substring_boost_multiple_hints():
    items = [
        {"name": "search_repos", "server_name": "github", "tags": "api,search", "vector_distance": 0.5},
    ]
    result = _apply_substring_boost(items, ["search", "api"])
    # Both hints match -> 0.5 * 2 * 2 = 2.0
    assert abs(result[0]["_boosted_score"] - 2.0) < 0.01


def test_input_schema_summary_truncates():
    long_schema = "a" * 500
    result = _input_schema_summary(long_schema)
    assert len(result) == 200


def test_input_schema_summary_empty():
    assert _input_schema_summary("") == ""
    assert _input_schema_summary(None) == ""


@respx.mock
@pytest.mark.asyncio
async def test_hybrid_search_no_index():
    """Returns empty list when tool index is None."""
    settings = make_settings()
    respx.post("http://fakeembed/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
        )
    )

    async with httpx.AsyncClient() as client:
        results = await hybrid_search(
            query="find search tools",
            top_k=5,
            settings=settings,
            tool_index=None,
            http_client=client,
        )

    assert results == []


@respx.mock
@pytest.mark.asyncio
async def test_hybrid_search_with_mock_index():
    """hybrid_search returns ToolSearchResult list from mock index."""
    settings = make_settings()
    respx.post("http://fakeembed/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
        )
    )

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[
        {
            "tool_id": "github:search_repos",
            "name": "search_repos",
            "description": "Search repos",
            "server_name": "github",
            "tags": "search,api",
            "input_schema": '{"type": "object"}',
            "score": 0.85,
            "vector_distance": 0.85,
        }
    ])

    async with httpx.AsyncClient() as client:
        results = await hybrid_search(
            query="search github repos",
            top_k=5,
            settings=settings,
            tool_index=mock_index,
            http_client=client,
        )

    assert len(results) == 1
    assert results[0].tool_id == "github:search_repos"
    assert results[0].score > 0


@respx.mock
@pytest.mark.asyncio
async def test_hybrid_search_filters_by_min_score():
    """Results below min_score are filtered out."""
    settings = make_settings()
    settings.search.min_score = 0.5  # set high threshold

    respx.post("http://fakeembed/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
        )
    )

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[
        {
            "tool_id": "github:search_repos",
            "name": "search_repos",
            "description": "Search repos",
            "server_name": "github",
            "tags": "",
            "input_schema": "",
            "vector_distance": 0.1,  # very low score
        }
    ])

    async with httpx.AsyncClient() as client:
        results = await hybrid_search(
            query="find something",
            top_k=5,
            settings=settings,
            tool_index=mock_index,
            http_client=client,
        )

    assert results == []


@respx.mock
@pytest.mark.asyncio
async def test_hybrid_search_hint_boosts_matching_tool():
    """@hint boosts tools with hint in name/tags over those without."""
    settings = make_settings()
    settings.search.min_score = 0.0

    respx.post("http://fakeembed/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 1024, "index": 0}]},
        )
    )

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[
        {
            "tool_id": "github:search_repos",
            "name": "search_repos",
            "description": "Search repos",
            "server_name": "github",
            "tags": "search",
            "input_schema": "",
            "vector_distance": 0.5,
        },
        {
            "tool_id": "github:create_issue",
            "name": "create_issue",
            "description": "Create issue",
            "server_name": "github",
            "tags": "write",
            "input_schema": "",
            "vector_distance": 0.5,
        },
    ])

    async with httpx.AsyncClient() as client:
        results = await hybrid_search(
            query="@search find tools",
            top_k=5,
            settings=settings,
            tool_index=mock_index,
            http_client=client,
        )

    # search_repos should rank higher due to hint boost
    assert len(results) >= 1
    if len(results) == 2:
        assert results[0].name == "search_repos"


@respx.mock
@pytest.mark.asyncio
async def test_hybrid_search_uses_cache(fakeredis_client):
    """Second call with same query hits cache, not vLLM."""
    settings = make_settings()

    call_count = 0

    def embedding_side_effect(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024, "index": 0}]})

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[])

    async with respx.mock:
        respx.post("http://fakeembed/v1/embeddings").mock(side_effect=embedding_side_effect)

        async with httpx.AsyncClient() as client:
            # First call
            await hybrid_search(
                query="test query cached",
                top_k=5,
                settings=settings,
                tool_index=mock_index,
                http_client=client,
                redis_client=fakeredis_client,
            )
            # Second call with same query
            await hybrid_search(
                query="test query cached",
                top_k=5,
                settings=settings,
                tool_index=mock_index,
                http_client=client,
                redis_client=fakeredis_client,
            )

    assert call_count == 1  # vLLM only called once
