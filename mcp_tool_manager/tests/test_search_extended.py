"""Extended search tests: filter building, ImportError fallback, vector-only."""
import pytest
import respx
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from mcp_tool_manager.services.search import (
    _build_hint_filter,
    hybrid_search,
    _execute_hybrid_query,
    _vector_only_search,
)
from mcp_tool_manager.config import Settings


def make_settings(**kwargs):
    import os
    overrides = {"EMBEDDING__API_URL": "http://fakeembed/v1/embeddings"}
    overrides.update(kwargs)
    old = {k: os.environ.get(k) for k in overrides}
    for k, v in overrides.items():
        os.environ[k] = v
    s = Settings()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return s


FAKE_VECTOR = b"\x00" * (1024 * 4)


def test_build_hint_filter_returns_none_for_no_hints():
    result = _build_hint_filter([])
    assert result is None


def test_build_hint_filter_single_hint():
    result = _build_hint_filter(["github"])
    assert result is not None


def test_build_hint_filter_multiple_hints():
    result = _build_hint_filter(["github", "api"])
    assert result is not None


def test_build_hint_filter_exception_returns_none():
    """If redisvl Tag raises, returns None gracefully."""
    with patch("redisvl.query.filter.Tag", side_effect=Exception("no redisvl")):
        result = _build_hint_filter(["github"])
    assert result is None


@respx.mock
@pytest.mark.asyncio
async def test_execute_hybrid_query_sets_filter():
    """HybridQuery.set_filter is called when hints present."""
    settings = make_settings()
    settings.search.min_score = 0.0

    mock_query = MagicMock()
    mock_query.set_filter = MagicMock()

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[])

    with patch("redisvl.query.HybridQuery", return_value=mock_query):
        await _execute_hybrid_query(
            clean_query="search repos",
            query_vector=FAKE_VECTOR,
            hints=["github"],
            hint_filter=MagicMock(),  # non-None filter
            top_k=10,
            settings=settings,
            tool_index=mock_index,
        )

    mock_query.set_filter.assert_called_once()


@respx.mock
@pytest.mark.asyncio
async def test_execute_hybrid_query_no_filter_when_no_hints():
    settings = make_settings()

    mock_query = MagicMock()
    mock_query.set_filter = MagicMock()
    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[])

    with patch("redisvl.query.HybridQuery", return_value=mock_query):
        await _execute_hybrid_query(
            clean_query="search repos",
            query_vector=FAKE_VECTOR,
            hints=[],
            hint_filter=None,
            top_k=10,
            settings=settings,
            tool_index=mock_index,
        )

    mock_query.set_filter.assert_not_called()


@respx.mock
@pytest.mark.asyncio
async def test_execute_hybrid_query_import_error_fallback():
    """ImportError on HybridQuery falls back to vector-only search."""
    settings = make_settings()
    settings.search.min_score = 0.0

    mock_vq = MagicMock()
    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[
        {"tool_id": "s:t", "name": "t", "description": "d",
         "server_name": "s", "tags": "", "input_schema": "", "vector_distance": 0.9}
    ])

    with patch("redisvl.query.HybridQuery", side_effect=ImportError("no hybrid")), \
         patch("redisvl.query.VectorQuery", return_value=mock_vq):
        result = await _execute_hybrid_query(
            clean_query="query",
            query_vector=FAKE_VECTOR,
            hints=[],
            hint_filter=None,
            top_k=5,
            settings=settings,
            tool_index=mock_index,
        )

    assert len(result) == 1


@respx.mock
@pytest.mark.asyncio
async def test_execute_hybrid_query_exception_fallback():
    """Exception during HybridQuery falls back to vector-only search."""
    settings = make_settings()

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[])

    mock_vq = MagicMock()
    with patch("redisvl.query.HybridQuery", side_effect=Exception("redis error")), \
         patch("redisvl.query.VectorQuery", return_value=mock_vq):
        result = await _execute_hybrid_query(
            clean_query="query",
            query_vector=FAKE_VECTOR,
            hints=[],
            hint_filter=None,
            top_k=5,
            settings=settings,
            tool_index=mock_index,
        )

    assert isinstance(result, list)


@respx.mock
@pytest.mark.asyncio
async def test_vector_only_search():
    settings = make_settings()

    mock_vq = MagicMock()
    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[
        {"tool_id": "s:t", "name": "t", "description": "d",
         "server_name": "s", "tags": "", "input_schema": ""}
    ])

    with patch("redisvl.query.VectorQuery", return_value=mock_vq):
        result = await _vector_only_search(FAKE_VECTOR, 5, settings, mock_index)

    assert len(result) == 1


@respx.mock
@pytest.mark.asyncio
async def test_hybrid_search_index_query_returns_none():
    """Tool index returns None (no results) — handled gracefully."""
    settings = make_settings()
    settings.search.min_score = 0.0

    respx.post("http://fakeembed/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024, "index": 0}]})
    )

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=None)

    async with httpx.AsyncClient() as client:
        results = await hybrid_search(
            query="test",
            top_k=5,
            settings=settings,
            tool_index=mock_index,
            http_client=client,
        )

    assert results == []


@respx.mock
@pytest.mark.asyncio
async def test_hybrid_search_index_exception():
    """Exception during search returns empty list."""
    settings = make_settings()

    respx.post("http://fakeembed/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024, "index": 0}]})
    )

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(side_effect=Exception("index down"))

    async with httpx.AsyncClient() as client:
        results = await hybrid_search(
            query="test",
            top_k=5,
            settings=settings,
            tool_index=mock_index,
            http_client=client,
        )

    assert results == []


@respx.mock
@pytest.mark.asyncio
async def test_hybrid_search_empty_clean_query_uses_hints():
    """When clean_query is empty after hint stripping, hints become the query."""
    settings = make_settings()
    settings.search.min_score = 0.0

    respx.post("http://fakeembed/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1] * 1024, "index": 0}]})
    )

    mock_index = AsyncMock()
    mock_index.query = AsyncMock(return_value=[])

    async with httpx.AsyncClient() as client:
        # Only hints, no clean query
        results = await hybrid_search(
            query="@github @search",
            top_k=5,
            settings=settings,
            tool_index=mock_index,
            http_client=client,
        )

    mock_index.query.assert_called_once()
    assert results == []
