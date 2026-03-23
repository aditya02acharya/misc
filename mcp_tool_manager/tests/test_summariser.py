"""Tests for AWS Bedrock summariser."""
import pytest
from unittest.mock import MagicMock, patch
from mcp_tool_manager.services.summariser import summarise_tool_result
from mcp_tool_manager.config import Settings


def make_bedrock_response(text: str) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "usage": {"inputTokens": 100, "outputTokens": 50},
    }


@pytest.mark.asyncio
async def test_summarise_returns_summary_and_gaps():
    settings = Settings()
    bedrock_client = MagicMock()
    bedrock_client.converse.return_value = make_bedrock_response(
        "Found 10 repos matching criteria.\nGaps: Missing language filter."
    )

    summary, gaps = await summarise_tool_result(
        tool_name="search_repos",
        arguments={"query": "python ml"},
        raw_result="repo1, repo2, repo3...",
        bedrock_client=bedrock_client,
        settings=settings,
    )

    assert "Found 10 repos" in summary
    assert "language filter" in gaps
    bedrock_client.converse.assert_called_once()


@pytest.mark.asyncio
async def test_summarise_no_gaps_marker():
    settings = Settings()
    bedrock_client = MagicMock()
    bedrock_client.converse.return_value = make_bedrock_response(
        "Tool executed successfully and returned results."
    )

    summary, gaps = await summarise_tool_result(
        tool_name="list_files",
        arguments={},
        raw_result="file1.py, file2.py",
        bedrock_client=bedrock_client,
        settings=settings,
    )

    assert summary == "Tool executed successfully and returned results."
    assert gaps == "No gaps identified."


@pytest.mark.asyncio
async def test_summarise_handles_exception():
    settings = Settings()
    bedrock_client = MagicMock()
    bedrock_client.converse.side_effect = Exception("Connection error")

    summary, gaps = await summarise_tool_result(
        tool_name="search_repos",
        arguments={},
        raw_result="some result",
        bedrock_client=bedrock_client,
        settings=settings,
    )

    # Should return fallback, not raise
    assert "search_repos" in summary
    assert "unavailable" in gaps


@pytest.mark.asyncio
async def test_summarise_truncates_long_result():
    settings = Settings()
    bedrock_client = MagicMock()
    bedrock_client.converse.return_value = make_bedrock_response("Summary.\nGaps: None.")

    long_result = "x" * 5000

    await summarise_tool_result(
        tool_name="big_tool",
        arguments={},
        raw_result=long_result,
        bedrock_client=bedrock_client,
        settings=settings,
    )

    call_args = bedrock_client.converse.call_args
    prompt_text = call_args[1]["messages"][0]["content"][0]["text"]
    # Result should be truncated in prompt
    assert len(prompt_text) < 4000
