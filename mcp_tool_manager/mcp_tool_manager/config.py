from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379/0"
    tool_index: str = "mcp_tools_idx"
    cache_index: str = "mcp_cache_idx"
    vector_dims: int = 1024


class EmbeddingSettings(BaseModel):
    api_url: str = "http://localhost:8080/v1/embeddings"
    api_key: str = ""
    model: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = 32
    timeout: float = 30.0


class SearchSettings(BaseModel):
    default_top_k: int = 10
    min_score: float = 0.3
    combination: str = "RRF"
    text_weight: float = 0.4
    vector_weight: float = 0.6
    name_boost: float = 3.0


class CacheSettings(BaseModel):
    ttl: int = 3600
    chunk_size: int = 1000
    chunk_overlap: int = 100


class BedrockSettings(BaseModel):
    region: str = "us-east-1"
    summary_model: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    max_tokens: int = 512
    endpoint_url: str | None = None


class RegistrySettings(BaseModel):
    api_url: str = "http://localhost:4000"
    api_key: str = ""
    sync_timeout: int = 120


class MCPServerSettings(BaseModel):
    default_transport: str = "streamable-http"
    connection_timeout: float = 30.0


class OtelSettings(BaseModel):
    enabled: bool = True
    endpoint: str = "http://localhost:6006"
    project_name: str = "mcp-tool-manager"
    api_key: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "mcp-tool-manager"
    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    redis: RedisSettings = RedisSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    search: SearchSettings = SearchSettings()
    cache: CacheSettings = CacheSettings()
    bedrock: BedrockSettings = BedrockSettings()
    registry: RegistrySettings = RegistrySettings()
    mcp_servers: MCPServerSettings = MCPServerSettings()
    otel: OtelSettings = OtelSettings()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
