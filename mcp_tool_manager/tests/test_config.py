"""Tests for configuration loading."""
import os
import pytest
from mcp_tool_manager.config import Settings, get_settings, RedisSettings, EmbeddingSettings


def test_default_settings():
    settings = Settings()
    assert settings.app_name == "mcp-tool-manager"
    assert settings.port == 8000
    assert settings.host == "0.0.0.0"
    assert settings.debug is False


def test_redis_defaults():
    settings = Settings()
    assert settings.redis.url == "redis://localhost:6379/0"
    assert settings.redis.tool_index == "mcp_tools_idx"
    assert settings.redis.cache_index == "mcp_cache_idx"
    assert settings.redis.vector_dims == 1024


def test_embedding_defaults():
    settings = Settings()
    assert settings.embedding.model == "BAAI/bge-large-en-v1.5"
    assert settings.embedding.batch_size == 32
    assert settings.embedding.timeout == 30.0


def test_search_defaults():
    settings = Settings()
    assert settings.search.default_top_k == 10
    assert settings.search.combination == "RRF"
    assert settings.search.min_score == 0.3


def test_cache_defaults():
    settings = Settings()
    assert settings.cache.ttl == 3600
    assert settings.cache.chunk_size == 1000
    assert settings.cache.chunk_overlap == 100


def test_bedrock_defaults():
    settings = Settings()
    assert settings.bedrock.region == "us-east-1"
    assert settings.bedrock.max_tokens == 512


def test_env_override(monkeypatch):
    monkeypatch.setenv("REDIS__URL", "redis://testhost:6379/1")
    monkeypatch.setenv("EMBEDDING__API_URL", "http://myvllm/v1/embeddings")
    monkeypatch.setenv("PORT", "9000")

    settings = Settings()
    assert settings.redis.url == "redis://testhost:6379/1"
    assert settings.embedding.api_url == "http://myvllm/v1/embeddings"
    assert settings.port == 9000


def test_nested_delimiter_works(monkeypatch):
    monkeypatch.setenv("SEARCH__DEFAULT_TOP_K", "20")
    monkeypatch.setenv("CACHE__TTL", "7200")

    settings = Settings()
    assert settings.search.default_top_k == 20
    assert settings.cache.ttl == 7200


def test_get_settings_singleton():
    import mcp_tool_manager.config as cfg
    cfg._settings = None  # reset
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
