from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ToolDoc(BaseModel):
    tool_id: str  # "{server_name}:{tool_name}"
    name: str
    description: str
    server_name: str
    server_description: str = ""
    tags: list[str] = Field(default_factory=list)
    input_schema: dict = Field(default_factory=dict)
    searchable_text: str = ""
    content_hash: str = ""
    embedding: bytes = b""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class CacheChunk(BaseModel):
    session_id: str
    tool_call_id: str
    tool_id: str
    chunk_idx: int
    text: str
    embedding: bytes = b""
    arguments_json: str = "{}"
    created_at: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class SyncResult(BaseModel):
    fetched: int = 0
    created: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    embedding_time_ms: float = 0.0
    total_time_ms: float = 0.0


class SyncJobStatus(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    stats: SyncResult | None = None
    error: str | None = None


class ToolSearchResult(BaseModel):
    tool_id: str
    name: str
    server_name: str
    description: str
    input_schema_summary: str
    score: float


class CallToolResult(BaseModel):
    tool_call_id: str
    summary: str
    gaps: str
    server: str
    tool_name: str


class FetchResult(BaseModel):
    chunks: list[CacheChunk]
    total_chunks: int
