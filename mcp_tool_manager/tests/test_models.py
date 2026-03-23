"""Tests for Pydantic data models."""
import pytest
from datetime import datetime
from mcp_tool_manager.models import (
    ToolDoc,
    CacheChunk,
    SyncResult,
    SyncJobStatus,
    ToolSearchResult,
    CallToolResult,
    FetchResult,
)


def test_tool_doc_defaults():
    doc = ToolDoc(
        tool_id="github:search_repos",
        name="search_repos",
        description="Search GitHub repositories",
        server_name="github",
    )
    assert doc.tool_id == "github:search_repos"
    assert doc.tags == []
    assert doc.input_schema == {}
    assert doc.embedding == b""
    assert isinstance(doc.created_at, datetime)


def test_tool_doc_with_tags():
    doc = ToolDoc(
        tool_id="github:search_repos",
        name="search_repos",
        description="Search GitHub repositories",
        server_name="github",
        tags=["search", "github", "api"],
    )
    assert "search" in doc.tags
    assert len(doc.tags) == 3


def test_cache_chunk():
    chunk = CacheChunk(
        session_id="sess123",
        tool_call_id="sess123:abc12345",
        tool_id="github:search_repos",
        chunk_idx=0,
        text="Some cached text",
    )
    assert chunk.session_id == "sess123"
    assert chunk.chunk_idx == 0
    assert chunk.created_at > 0


def test_sync_result_defaults():
    result = SyncResult()
    assert result.fetched == 0
    assert result.created == 0
    assert result.embedding_time_ms == 0.0


def test_sync_result_values():
    result = SyncResult(fetched=150, created=2, updated=5, deleted=1, unchanged=142)
    assert result.fetched == 150
    assert result.created + result.updated + result.deleted + result.unchanged == 150


def test_sync_job_status_running():
    job = SyncJobStatus(job_id="sync_abc123", status="running")
    assert job.status == "running"
    assert job.completed_at is None
    assert job.stats is None


def test_sync_job_status_completed():
    stats = SyncResult(fetched=10, created=5, unchanged=5)
    job = SyncJobStatus(
        job_id="sync_abc123",
        status="completed",
        completed_at=datetime.utcnow(),
        stats=stats,
    )
    assert job.stats.fetched == 10


def test_tool_search_result():
    result = ToolSearchResult(
        tool_id="github:search_repos",
        name="search_repos",
        server_name="github",
        description="Search repos",
        input_schema_summary="query: str, language: str",
        score=0.87,
    )
    assert result.score == 0.87


def test_call_tool_result():
    result = CallToolResult(
        tool_call_id="sess1:abc12345",
        summary="Found 10 repos matching criteria",
        gaps="Missing language filter results",
        server="github",
        tool_name="search_repos",
    )
    assert result.tool_call_id == "sess1:abc12345"


def test_fetch_result():
    chunks = [
        CacheChunk(
            session_id="s1",
            tool_call_id="s1:abc",
            tool_id="github:search",
            chunk_idx=0,
            text="chunk text",
        )
    ]
    result = FetchResult(chunks=chunks, total_chunks=1)
    assert result.total_chunks == 1
    assert len(result.chunks) == 1
