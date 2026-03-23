"""Tests for MCP tool functions: find_tools, call_tool, fetch_cached_data."""
import asyncio
import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from mcp_tool_manager.config import Settings
from mcp_tool_manager.models import ToolSearchResult, FetchResult, CacheChunk


# ───────────────────────────── find_tools ─────────────────────────────

@pytest.mark.asyncio
async def test_find_tools_returns_results():
    mock_results = [
        ToolSearchResult(
            tool_id="github:search_repos",
            name="search_repos",
            server_name="github",
            description="Search repos",
            input_schema_summary="query: str",
            score=0.9,
        )
    ]

    with patch("mcp_tool_manager.tools.find.get_settings", return_value=Settings()), \
         patch("mcp_tool_manager.tools.find.get_tool_index", return_value=None), \
         patch("mcp_tool_manager.tools.find.get_httpx", return_value=MagicMock()), \
         patch("mcp_tool_manager.tools.find.get_redis", return_value=None), \
         patch("mcp_tool_manager.tools.find.hybrid_search", new_callable=AsyncMock, return_value=mock_results):

        from mcp_tool_manager.tools.find import find_tools
        result = await find_tools(query="search github", top_k=5)

    assert len(result) == 1
    assert result[0]["tool_id"] == "github:search_repos"
    assert result[0]["score"] == 0.9


@pytest.mark.asyncio
async def test_find_tools_with_server_filter():
    with patch("mcp_tool_manager.tools.find.get_settings", return_value=Settings()), \
         patch("mcp_tool_manager.tools.find.get_tool_index", return_value=None), \
         patch("mcp_tool_manager.tools.find.get_httpx", return_value=MagicMock()), \
         patch("mcp_tool_manager.tools.find.get_redis", return_value=None), \
         patch("mcp_tool_manager.tools.find.hybrid_search", new_callable=AsyncMock, return_value=[]) as mock_search:

        from mcp_tool_manager.tools.find import find_tools
        await find_tools(query="find tools", top_k=3, server_filter="github")

    _, kwargs = mock_search.call_args
    assert kwargs.get("server_filter") == "github"


@pytest.mark.asyncio
async def test_find_tools_empty_results():
    with patch("mcp_tool_manager.tools.find.get_settings", return_value=Settings()), \
         patch("mcp_tool_manager.tools.find.get_tool_index", return_value=None), \
         patch("mcp_tool_manager.tools.find.get_httpx", return_value=MagicMock()), \
         patch("mcp_tool_manager.tools.find.get_redis", return_value=None), \
         patch("mcp_tool_manager.tools.find.hybrid_search", new_callable=AsyncMock, return_value=[]):

        from mcp_tool_manager.tools.find import find_tools
        result = await find_tools(query="obscure query")

    assert result == []


# ───────────────────────────── call_tool ──────────────────────────────

@pytest.mark.asyncio
async def test_call_tool_success():
    mock_raw_result = MagicMock()
    mock_raw_result.content = [MagicMock(text="result text")]

    mock_queue = asyncio.Queue(maxsize=10)
    mock_bedrock = MagicMock()
    mock_bedrock.converse.return_value = {
        "output": {"message": {"content": [{"text": "Summary.\nGaps: None."}]}}
    }

    with patch("mcp_tool_manager.tools.call.get_settings", return_value=Settings()), \
         patch("mcp_tool_manager.tools.call.get_redis", return_value=AsyncMock()), \
         patch("mcp_tool_manager.tools.call.get_bedrock", return_value=mock_bedrock), \
         patch("mcp_tool_manager.tools.call.get_cache_queue", return_value=mock_queue), \
         patch(
             "mcp_tool_manager.tools.call.call_tool_with_validation",
             new_callable=AsyncMock,
             return_value=("search_repos", mock_raw_result),
         ):

        from mcp_tool_manager.tools.call import call_tool
        result = await call_tool(
            session_id="sess1",
            tool_id="github:search_repos",
            arguments={"query": "python"},
        )

    assert "tool_call_id" in result
    assert result["tool_call_id"].startswith("sess1:")
    assert result["server"] == "github"
    assert result["tool_name"] == "search_repos"
    assert "summary" in result
    assert "gaps" in result


@pytest.mark.asyncio
async def test_call_tool_cache_queue_full():
    """Full queue: drops cache write without raising."""
    mock_raw_result = MagicMock()
    mock_raw_result.content = []

    # Full queue
    mock_queue = asyncio.Queue(maxsize=1)
    mock_queue.put_nowait(("dummy",))

    mock_bedrock = MagicMock()
    mock_bedrock.converse.return_value = {
        "output": {"message": {"content": [{"text": "Summary.\nGaps: None."}]}}
    }

    with patch("mcp_tool_manager.tools.call.get_settings", return_value=Settings()), \
         patch("mcp_tool_manager.tools.call.get_redis", return_value=AsyncMock()), \
         patch("mcp_tool_manager.tools.call.get_bedrock", return_value=mock_bedrock), \
         patch("mcp_tool_manager.tools.call.get_cache_queue", return_value=mock_queue), \
         patch(
             "mcp_tool_manager.tools.call.call_tool_with_validation",
             new_callable=AsyncMock,
             return_value=("some_tool", mock_raw_result),
         ):

        from mcp_tool_manager.tools.call import call_tool
        # Should not raise even though queue is full
        result = await call_tool(
            session_id="sess2",
            tool_id="server:some_tool",
            arguments={},
        )

    assert "tool_call_id" in result


@pytest.mark.asyncio
async def test_call_tool_validation_error_propagates():
    mock_queue = asyncio.Queue(maxsize=10)

    with patch("mcp_tool_manager.tools.call.get_settings", return_value=Settings()), \
         patch("mcp_tool_manager.tools.call.get_redis", return_value=AsyncMock()), \
         patch("mcp_tool_manager.tools.call.get_bedrock", return_value=MagicMock()), \
         patch("mcp_tool_manager.tools.call.get_cache_queue", return_value=mock_queue), \
         patch(
             "mcp_tool_manager.tools.call.call_tool_with_validation",
             new_callable=AsyncMock,
             side_effect=ValueError("Tool not found"),
         ):

        from mcp_tool_manager.tools.call import call_tool
        with pytest.raises(ValueError, match="Tool not found"):
            await call_tool(
                session_id="sess3",
                tool_id="ghost:tool",
                arguments={},
            )


# ───────────────────────── fetch_cached_data ──────────────────────────

@pytest.mark.asyncio
async def test_fetch_cached_data_by_tool_call_id():
    mock_chunk = CacheChunk(
        session_id="sess1", tool_call_id="sess1:abc", tool_id="g:t",
        chunk_idx=0, text="data"
    )
    mock_result = FetchResult(chunks=[mock_chunk], total_chunks=1)

    with patch("mcp_tool_manager.tools.fetch.get_settings", return_value=Settings()), \
         patch("mcp_tool_manager.tools.fetch.get_redis", return_value=AsyncMock()), \
         patch("mcp_tool_manager.tools.fetch.get_cache_index", return_value=None), \
         patch("mcp_tool_manager.tools.fetch.get_httpx", return_value=MagicMock()), \
         patch(
             "mcp_tool_manager.tools.fetch.fetch_by_tool_call_id",
             new_callable=AsyncMock,
             return_value=mock_result,
         ) as mock_fn:

        from mcp_tool_manager.tools.fetch import fetch_cached_data
        result = await fetch_cached_data(session_id="sess1", tool_call_id="sess1:abc")

    mock_fn.assert_called_once_with("sess1", "sess1:abc", ANY)
    assert result["total_chunks"] == 1
    assert len(result["chunks"]) == 1


@pytest.mark.asyncio
async def test_fetch_cached_data_by_query():
    mock_result = FetchResult(chunks=[], total_chunks=0)

    with patch("mcp_tool_manager.tools.fetch.get_settings", return_value=Settings()), \
         patch("mcp_tool_manager.tools.fetch.get_redis", return_value=AsyncMock()), \
         patch("mcp_tool_manager.tools.fetch.get_cache_index", return_value=None), \
         patch("mcp_tool_manager.tools.fetch.get_httpx", return_value=MagicMock()), \
         patch(
             "mcp_tool_manager.tools.fetch.fetch_by_query",
             new_callable=AsyncMock,
             return_value=mock_result,
         ) as mock_fn:

        from mcp_tool_manager.tools.fetch import fetch_cached_data
        result = await fetch_cached_data(session_id="sess1", query="find repos", top_k=3)

    mock_fn.assert_called_once()
    assert result["total_chunks"] == 0


@pytest.mark.asyncio
async def test_fetch_cached_data_all_session():
    mock_result = FetchResult(chunks=[], total_chunks=0)

    with patch("mcp_tool_manager.tools.fetch.get_settings", return_value=Settings()), \
         patch("mcp_tool_manager.tools.fetch.get_redis", return_value=AsyncMock()), \
         patch("mcp_tool_manager.tools.fetch.get_cache_index", return_value=None), \
         patch("mcp_tool_manager.tools.fetch.get_httpx", return_value=MagicMock()), \
         patch(
             "mcp_tool_manager.tools.fetch.fetch_all_session",
             new_callable=AsyncMock,
             return_value=mock_result,
         ) as mock_fn:

        from mcp_tool_manager.tools.fetch import fetch_cached_data
        result = await fetch_cached_data(session_id="sess1")

    mock_fn.assert_called_once_with("sess1", 5, ANY)
    assert result["total_chunks"] == 0
