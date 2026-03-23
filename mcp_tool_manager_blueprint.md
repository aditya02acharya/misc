# MCP Tool Manager — Design Blueprint

> **Purpose**: A sidecar MCP server that helps LLM agents manage large numbers of MCP tools without overburdening the context window. Three tools: `find_tools`, `call_tool`, `fetch_cached_data`.
>
> **Design Principles**: Minimum lines of code. No premature abstraction. Use libraries that do the heavy lifting. Every module earns its existence.

---

## 1. Architecture Overview

```
                         Agent (LLM)
                             │
                    ┌────────┴────────┐
                    │  MCP Tool Mgr   │  ← FastMCP (streamable-http)
                    │  (sidecar)      │
                    ├─────────────────┤
                    │ find_tools      │─── hybrid search ──┐
                    │ call_tool       │─── execute + cache  │
                    │ fetch_cached    │─── retrieve context │
                    ├─────────────────┤                     │
                    │ REST API        │                     │
                    │  POST /sync     │                     │
                    │  GET /metrics   │                     │
                    │  GET /health    │                     │
                    └────────┬────────┘                     │
                             │                              │
              ┌──────────────┴──────────────┐               │
              │         Redis 8.4+          │◄──────────────┘
              │  ┌──────────┬────────────┐  │
              │  │ Tool Idx │ Cache Idx  │  │
              │  │ (HNSW)   │ (HNSW)    │  │
              │  │ + BM25   │ + Session  │  │
              │  └──────────┴────────────┘  │
              └─────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │        vLLM (REST)          │  ← embedding service
              └─────────────────────────────┘
              ┌──────────────┴──────────────┐
              │   Phoenix Arize (OTEL)      │  ← telemetry collector
              └─────────────────────────────┘
              ┌──────────────┴──────────────┐
              │   LiteLLM (REST)            │  ← server registry
              └─────────────────────────────┘
```

**Key simplification vs reference design**: No AWS Secrets Manager, no SSL context builder, no multi-vector indexing. Secrets via env vars + pydantic-settings. Single combined embedding per tool + native Redis hybrid search (FT.HYBRID / RRF). `redisvl` handles all vector operations.

---

## 2. Technology Stack

| Component | Library | Version | Why |
|---|---|---|---|
| MCP Server | `fastmcp` (PrefectHQ) | ≥3.1 | Decorator-based tools, built-in progress/sampling/elicitation, streamable-http |
| REST API | `fastapi` + `uvicorn` | latest | Sync/metrics/health endpoints; mount alongside FastMCP |
| Redis Client | `redisvl` | ≥0.16 | `HybridQuery` (RRF built-in), `SearchIndex`, `SemanticCache`; wraps `redis[hiredis]` |
| Embeddings | `httpx` | latest | Async REST calls to external vLLM `/v1/embeddings` |
| Settings | `pydantic-settings` v2 | latest | Nested model with `env_nested_delimiter="__"` |
| Validation | `jsonschema` | latest | Validate tool call arguments against stored input_schema |
| Telemetry | `phoenix.otel` + `openinference-semconv` | latest | Phoenix Arize-aware OTEL wrapper; OpenInference semantic conventions |
| Summariser | `boto3` (Bedrock Runtime) | latest | AWS Bedrock `converse` API for fast small-model summarisation |
| MCP Client | `mcp` SDK | latest | `ClientSession` to call downstream MCP servers |
| Background | `asyncio.Queue` | stdlib | Non-blocking cache writes; no Celery/RQ needed |
| Hashing | `hashlib` | stdlib | SHA-256 for incremental sync diff |

### Notable library choices

- **`redisvl.query.HybridQuery`** — Redis 8.4+ native `FT.HYBRID` command with built-in RRF and LINEAR fusion. Eliminates manual score fusion code entirely.
- **`redisvl.query.AggregateHybridQuery`** — Fallback for Redis < 8.4. Same interface.
- **`fastmcp.Context`** — Gives access to `ctx.report_progress()`, `ctx.session.create_message()` (sampling), and `ctx.session.elicit()` (elicitation). All MCP features pass through without extra code.
- **`phoenix.otel.register()`** — One-liner OTEL setup. Auto-discovers OpenInference instrumentors. Reads `PHOENIX_COLLECTOR_ENDPOINT` and `PHOENIX_API_KEY` from env.

---

## 3. Project Structure

```
mcp_tool_manager/
├── __init__.py
├── main.py              # Entrypoint: mount FastMCP + FastAPI, lifespan
├── config.py            # All pydantic-settings models
├── telemetry.py         # phoenix.otel register + custom metrics helper
├── dependencies.py      # Shared singletons: redis index, httpx client, queues
│
├── tools/
│   ├── __init__.py
│   ├── find.py          # find_tools MCP tool
│   ├── call.py          # call_tool MCP tool
│   └── fetch.py         # fetch_cached_data MCP tool
│
├── services/
│   ├── __init__.py
│   ├── embedding.py     # vLLM embedding client (embed / embed_batch)
│   ├── registry.py      # LiteLLM registry client + sync job logic
│   ├── search.py        # Build HybridQuery, handle @hints, RRF
│   ├── executor.py      # MCP client session mgmt, call downstream servers
│   ├── summariser.py    # boto3 Bedrock converse wrapper
│   └── cache.py         # Async cache write: chunk → embed → store with TTL
│
├── models.py            # All pydantic data models (ToolDoc, CacheChunk, SyncResult, etc.)
│
├── api/
│   ├── __init__.py
│   ├── sync.py          # POST /v1/sync, GET /v1/sync/{job_id}
│   ├── metrics.py       # GET /v1/metrics (flush OTEL counters)
│   └── health.py        # GET /v1/health, GET /v1/ready
│
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml   # redis, vllm, phoenix, app
```

**~16 files total.** Each file has a single responsibility. No `interfaces/` folder, no `abstractions/`. If you need to swap a component, swap the module.

---

## 4. Configuration (Pydantic Settings)

### Pattern

One `Settings` root model. Component sub-models use `env_prefix` via nested delimiter `__`.

```
# config.py structure (NO CODE — just the schema)

Settings (root, .env file)
├── app_name: str = "mcp-tool-manager"
├── debug: bool = False
├── log_level: str = "INFO"
├── host: str = "0.0.0.0"
├── port: int = 8000
│
├── redis: RedisSettings         ← prefix REDIS__
│   ├── url: str = "redis://localhost:6379/0"
│   ├── tool_index: str = "mcp_tools_idx"
│   ├── cache_index: str = "mcp_cache_idx"
│   └── vector_dims: int = 1024
│
├── embedding: EmbeddingSettings ← prefix EMBEDDING__
│   ├── api_url: str             (required, vLLM endpoint)
│   ├── api_key: str = ""
│   ├── model: str = "BAAI/bge-large-en-v1.5"
│   ├── batch_size: int = 32
│   └── timeout: float = 30.0
│
├── search: SearchSettings       ← prefix SEARCH__
│   ├── default_top_k: int = 10
│   ├── min_score: float = 0.3
│   ├── combination: str = "RRF"      # or "LINEAR"
│   ├── text_weight: float = 0.4      # only for LINEAR
│   ├── vector_weight: float = 0.6    # only for LINEAR
│   └── name_boost: float = 3.0       # BM25 field weight for name
│
├── cache: CacheSettings         ← prefix CACHE__
│   ├── ttl: int = 3600
│   ├── chunk_size: int = 1000
│   └── chunk_overlap: int = 100
│
├── bedrock: BedrockSettings     ← prefix BEDROCK__
│   ├── region: str = "us-east-1"
│   ├── summary_model: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
│   ├── max_tokens: int = 512
│   └── endpoint_url: str | None = None   # for localstack/testing
│
├── registry: RegistrySettings   ← prefix REGISTRY__
│   ├── api_url: str = "http://localhost:4000"
│   ├── api_key: str = ""
│   └── sync_timeout: int = 120
│
├── mcp_servers: MCPServerSettings ← prefix MCP_SERVERS__
│   ├── default_transport: str = "streamable-http"
│   └── connection_timeout: float = 30.0
│
└── otel: OtelSettings           ← prefix OTEL__
    ├── enabled: bool = True
    ├── endpoint: str = "http://localhost:6006"
    ├── project_name: str = "mcp-tool-manager"
    └── api_key: str = ""
```

### Environment variables

```bash
REDIS__URL=redis://redis:6379/0
EMBEDDING__API_URL=https://vllm.internal/v1/embeddings
EMBEDDING__API_KEY=sk-xxx
BEDROCK__REGION=us-east-1
BEDROCK__SUMMARY_MODEL=us.anthropic.claude-3-5-haiku-20241022-v1:0
REGISTRY__API_URL=http://litellm:4000
OTEL__ENDPOINT=http://phoenix:6006
OTEL__API_KEY=your-phoenix-key
# AWS credentials via standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN or IAM role
```

---

## 5. MCP Tool Designs

### 5.1 `find_tools`

**Purpose**: Agent calls this to discover relevant tools for a query.

**Input Schema** (what the agent sees):

```
find_tools(
    query: str,        # "search github repos" or "@github search repos" or "@search find code"
    top_k: int = 10,
    server_filter: str | None = None
)
```

**Tool Description** (in MCP listing — prompt engineering the agent):
```
Discover available tools. Describe WHAT you need to accomplish, not HOW.
Use @hint to narrow results — hint matches against server names, tool names, 
and tags (e.g., "@github", "@search_repos", "@api", "@database").
Multiple @hints supported: "@github @search find code repos".
Returns ranked tools with name, description, server, and input schema summary.
```

**Internal Flow**:

```
1. Parse @hints from query:
   - Extract all @tokens: re.findall(r'@(\S+)', query)
   - Strip @tokens from query → clean_query for semantic search
   - hints = ["github", "search_repos", "api"]  (can be zero or many)

2. Embed clean_query via vLLM → query_vector (cache in Redis, 5min TTL)

3. Build hint filter (if hints present):
   - For each hint, build OR match across three TAG fields:
     Tag("server_name") == hint | Tag("name_tag") == hint | Tag("tags") == hint
   - TAG fields are exact match. For substring/partial matching:
     inject hint into the BM25 text query with boost, e.g., "(@name:{hint})=>{$weight:5}"
   - Multiple hints combined with AND (all must match something)
   - Example: @github @api → tool must relate to "github" AND "api"
     across any combination of server_name, tool name, or tags

4. Build redisvl HybridQuery:
   - text = clean_query + boosted hint terms (appended to BM25 query)
   - vector = query_vector, vector_field = "embedding"
   - combination_method = settings.search.combination (RRF or LINEAR)
   - filter_expression = tag-level hint filter (if any exact TAG matches)
   - num_results = top_k * 2 (over-fetch for post-processing)

5. Execute index.query(hybrid_query)

6. Post-process:
   - Boost results where any hint appears as SUBSTRING in:
     tool name, server_name, tags, or description
   - 2x score multiplier per matching hint
   - This handles fuzzy: @search boosts "search_repos",
     "code_search", "global_search" equally
   - Apply min_score threshold
   - Truncate to top_k

7. Return list of {tool_id, name, server, description, input_schema_summary}
```

**Search Index Schema** (redisvl YAML):

```yaml
index:
  name: mcp_tools_idx
  prefix: "tool:"
  storage_type: hash
fields:
  - name: name
    type: text
    attrs: { weight: 3.0 }
  - name: name_tag
    type: tag                    # duplicate of name as TAG for @hint exact filtering
    attrs: { separator: "," }
  - name: description
    type: text
    attrs: { weight: 2.0 }
  - name: searchable_text
    type: text
    attrs: { weight: 1.0 }
  - name: server_name
    type: tag
    attrs: { separator: "," }
  - name: tags
    type: tag
    attrs: { separator: "," }
  - name: embedding
    type: vector
    attrs:
      algorithm: hnsw
      dims: 1024
      distance_metric: cosine
      datatype: float32
  - name: input_schema
    type: text
    attrs: { no_index: true }   # stored but not searchable
  - name: content_hash
    type: text
    attrs: { no_index: true }
```

**Embedding Text Composition** (single combined embedding per tool):

```
Tool: {name}
Function: {name}
Description: {description}
Server: {server_name} — {server_description}
Capabilities: {tags}
Parameters: {first 5 param names and descriptions}
```

This keeps it to **one embedding call per tool** during sync and **one embedding call per query** at search time. The BM25 side of the hybrid search handles field-level boosting (name weight 3.0, description 2.0).

**Accuracy Strategy for vague/overlapping tools**:

- BM25 field weighting handles exact term matches (name=3x, desc=2x)
- Semantic embedding handles intent matching
- RRF fusion naturally handles different score scales
- `@hint` provides multi-field narrowing — matches against server name, tool name, AND tags simultaneously. Multiple hints compose with AND for precise filtering (e.g., `@github @api` = GitHub server's API-tagged tools)
- Substring post-boost catches partial matches the TAG filter misses (e.g., `@search` boosts `search_repos`, `code_search`, `search_issues`)
- Query embedding cache (Redis key `qemb:{sha256(query)[:16]}`, TTL 5min) eliminates vLLM latency on repeated/similar queries

### 5.2 `call_tool`

**Purpose**: Execute an MCP tool on a downstream server, validate inputs, cache results.

**Input Schema**:

```
call_tool(
    session_id: str,      # session context for accumulating data
    tool_id: str,         # "server_name:tool_name"
    arguments: dict       # the tool's arguments
)
```

**Tool Description**:
```
Execute a discovered tool. Provide session_id to accumulate context across calls.
tool_id format: "server_name:tool_name" (from find_tools results).
arguments: must match the tool's input schema.
Returns: summary of results, gaps identified, and a tool_call_id for fetching raw data.
```

**Internal Flow**:

```
1. Parse tool_id → server_name, tool_name
2. Fetch tool doc from Redis  →  get input_schema
3. Validate arguments against input_schema (jsonschema.validate)
4. Report progress: ctx.report_progress(0.1, 1.0, "Connecting...")
5. Get/create MCP ClientSession for server_name:
   - Read server endpoint from registry or config
   - Use mcp SDK ClientSession with streamable-http transport
   - Connection pool: one session per server, reused
6. Execute: result = await session.call_tool(tool_name, arguments)
7. Report progress: ctx.report_progress(0.5, 1.0, "Processing results...")
8. Generate summary via AWS Bedrock (boto3 converse API):
   - Prompt: "Given tool {name} called with {args}, summarise this result.
     Identify any gaps or missing information. Be concise."
   - Model: settings.bedrock.summary_model (e.g., Claude Haiku)
   - bedrock_client.converse(modelId=..., messages=[...], inferenceConfig={maxTokens: 512})
   - This gives the agent enough context to decide next steps
9. Generate tool_call_id = f"{session_id}:{uuid4().hex[:8]}"
10. Queue async cache write (don't block return):
    - Put (session_id, tool_call_id, tool_id, arguments, raw_result)
      onto asyncio.Queue
11. Return: { tool_call_id, summary, gaps, server, tool_name }
```

**Async Cache Writer** (background task consuming from queue):

```
Forever loop:
  1. Dequeue item
  2. Serialize raw_result to text
  3. If len(text) > chunk_size:
     - Split into overlapping chunks (chunk_size, chunk_overlap)
  4. Batch embed all chunks via vLLM
  5. For each chunk:
     - Store in Redis as hash: cache:{session_id}:{tool_call_id}:{chunk_idx}
     - Fields: text, embedding, tool_id, arguments_json, created_at
     - Set TTL = settings.cache.ttl
  6. Record metrics: cache write latency, chunk count
```

**MCP Feature Pass-through**:

- **Progress**: `ctx.report_progress()` at connection, execution, and processing stages
- **Sampling**: If the downstream server requests sampling (via `create_message`), the call_tool handler passes it through to the agent via `ctx.session.create_message()`. This requires the agent's MCP client to support sampling callbacks.
- **Elicitation**: Similarly, if the downstream server needs user input, `ctx.session.elicit()` passes the Pydantic schema back to the agent/user.

### 5.3 `fetch_cached_data`

**Purpose**: Retrieve accumulated context for a session.

**Input Schema**:

```
fetch_cached_data(
    session_id: str,
    tool_call_id: str | None = None,    # filter to specific call
    query: str | None = None,           # semantic search within session
    top_k: int = 5
)
```

**Tool Description**:
```
Retrieve cached context from previous tool calls in this session.
- No filters: returns all cached chunks for the session (most recent first).
- tool_call_id: returns chunks from a specific call.
- query: semantic search across all session data, returns top_k matches.
```

**Internal Flow**:

```
Case 1: tool_call_id provided
  → Redis SCAN pattern: cache:{session_id}:{tool_call_id}:*
  → Return all chunks in order

Case 2: query provided
  → Embed query via vLLM
  → VectorQuery on cache_index with Tag filter session_id={session_id}
  → Return top_k chunks

Case 3: no filters
  → Redis SCAN pattern: cache:{session_id}:*
  → Sort by created_at desc
  → Return all (with pagination via top_k)
```

**Cache Index Schema**:

```yaml
index:
  name: mcp_cache_idx
  prefix: "cache:"
  storage_type: hash
fields:
  - name: session_id
    type: tag
  - name: tool_call_id
    type: tag
  - name: tool_id
    type: tag
  - name: text
    type: text
  - name: embedding
    type: vector
    attrs:
      algorithm: hnsw
      dims: 1024
      distance_metric: cosine
      datatype: float32
  - name: created_at
    type: numeric
    attrs: { sortable: true }
```

---

## 6. REST Endpoints

### POST /v1/sync

Trigger a full resync of tools from LiteLLM registry.

```
Request:  POST /v1/sync  { "force": false }
Response: { "job_id": "sync_abc123", "status": "running" }
```

**Sync Job Flow**:

```
1. Check if job already running (Redis key sync:lock with TTL 300s)
   → If locked, return existing job_id + status
2. Acquire lock, generate job_id
3. Store job status in Redis: sync:job:{job_id} = { status: "running", started_at }
4. Background task:
   a. Fetch all servers from LiteLLM registry
   b. For each server, fetch tools
   c. Compute content_hash (SHA-256 of name+desc+schema+server)
   d. Fetch existing hashes from Redis (pipeline HGET)
   e. Diff: new / updated / deleted / unchanged
   f. Batch embed only new+updated tools
   g. Store in Redis with content_hash
   h. Delete removed tools
   i. Update job status: { status: "completed", stats: SyncResult }
5. Release lock
```

**Job status TTL**: 1 hour after completion.

### GET /v1/sync/{job_id}

Poll sync job status.

```
Response: {
  "job_id": "sync_abc123",
  "status": "completed",       # running | completed | failed
  "stats": {
    "fetched": 150, "created": 2, "updated": 5,
    "deleted": 1, "unchanged": 142,
    "embedding_time_ms": 125.5, "total_time_ms": 350.2
  }
}
```

### GET /v1/metrics

Flush custom OTEL metrics. Returns Prometheus-compatible text.

### GET /v1/health

```
Response: {
  "status": "healthy",
  "redis": "ok",
  "vllm": "ok",
  "registry": "ok",
  "tool_count": 150,
  "last_sync": "2026-03-22T10:00:00Z"
}
```

---

## 7. Telemetry (Phoenix Arize + OpenInference)

### Setup (one-liner in `telemetry.py`)

```
# Conceptual — not code, just the pattern

phoenix.otel.register(
    project_name = settings.otel.project_name,
    auto_instrument = True      # auto-discovers openinference instrumentors
)
# That's it. Reads PHOENIX_COLLECTOR_ENDPOINT and PHOENIX_API_KEY from env.
```

### Phoenix / OpenInference Custom Attributes

Phoenix uses OpenInference semantic conventions. Key attributes to set on spans:

| Attribute | OpenInference Key | Where |
|---|---|---|
| Input query | `input.value` | find_tools span |
| Output tools | `output.value` | find_tools span |
| LLM model | `llm.model_name` | summary call span |
| Token count (input) | `llm.token_count.prompt` | summary call span |
| Token count (output) | `llm.token_count.completion` | summary call span |
| Retrieval query | `retrieval.query` | find_tools, fetch_cached |
| Retrieval docs | `retrieval.documents` | find_tools results |
| Tool name | `tool.name` | call_tool span |
| Tool parameters | `tool.parameters` | call_tool span |
| Session ID | `session.id` | all spans |
| MCP server | `metadata` (JSON) | call_tool span |

### Custom Metric Definitions

```
# Conceptual metric registry — implement as thin wrapper over otel MeterProvider

Counters:
  mcp.tools.total                    # gauge: total indexed tools
  mcp.tool_calls.count               # counter: by server, tool, status
  mcp.tool_calls.errors              # counter: by server, error_type
  mcp.search.count                   # counter: find_tools invocations
  mcp.cache.writes                   # counter: chunks written
  mcp.cache.reads                    # counter: chunks read
  mcp.sync.runs                      # counter: sync job runs

Histograms:
  mcp.tool_calls.latency_ms          # the ACTUAL downstream MCP call, not call_tool
  mcp.search.latency_ms              # find_tools end-to-end
  mcp.embedding.latency_ms           # vLLM call duration
  mcp.summary.latency_ms             # LLM summary call duration
  mcp.cache.write_latency_ms         # async cache write

Gauges:
  mcp.tools.total                    # current tool count
  mcp.cache.queue_depth              # pending cache writes
  mcp.sessions.active                # active session count
```

**Flexibility pattern**: Define metrics in a dict/registry so adding new ones is a one-liner:

```
# Pseudocode pattern
METRICS = {
    "mcp.tool_calls.latency_ms": ("histogram", "ms", "Downstream MCP call latency"),
    "mcp.search.count": ("counter", "1", "Search invocations"),
    # add more here — one line each
}
```

### GET /v1/metrics endpoint

Reads from the OTEL `MeterProvider`, formats as Prometheus exposition format, returns text. This is a pull-based complement to the push-based OTEL export.

---

## 8. Implementation Sessions

Each session is designed to fit within an LLM context window. Earlier sessions are prerequisites for later ones. Each session produces working, testable code.

---

### Session 1: Foundation

**Goal**: Project skeleton, config, Redis connection, embedding client, telemetry init.

**Files to create**:
- `pyproject.toml` — all dependencies
- `mcp_tool_manager/__init__.py`
- `mcp_tool_manager/config.py` — all Settings models
- `mcp_tool_manager/telemetry.py` — `phoenix.otel.register()` + metric registry helper
- `mcp_tool_manager/dependencies.py` — singleton holders (redis, httpx, boto3 bedrock client, queues)
- `mcp_tool_manager/models.py` — all pydantic data models
- `mcp_tool_manager/services/embedding.py` — vLLM client
- `mcp_tool_manager/services/summariser.py` — boto3 Bedrock converse wrapper (thin: ~20 lines)
- `mcp_tool_manager/main.py` — bare FastMCP + FastAPI mount, lifespan (startup/shutdown)

**Dependencies** (pyproject.toml):
```toml
dependencies = [
    "fastmcp>=3.1",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "redisvl>=0.16",
    "redis[hiredis]>=5.2",
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "jsonschema>=4.23",
    "boto3>=1.35",
    "arize-phoenix-otel>=0.13",
    "openinference-semantic-conventions>=0.1",
    "opentelemetry-api>=1.30",
    "opentelemetry-sdk>=1.30",
]
```

**Pydantic Models** (models.py):
```
ToolDoc:          tool_id, name, description, server_name, server_description,
                  tags, input_schema (dict), searchable_text, content_hash,
                  embedding (bytes), created_at, updated_at

CacheChunk:       session_id, tool_call_id, tool_id, chunk_idx,
                  text, embedding (bytes), arguments_json, created_at

SyncResult:       fetched, created, updated, deleted, unchanged,
                  embedding_time_ms, total_time_ms

SyncJobStatus:    job_id, status (running|completed|failed),
                  started_at, completed_at, stats (SyncResult|None),
                  error (str|None)

ToolSearchResult: tool_id, name, server_name, description,
                  input_schema_summary, score

CallToolResult:   tool_call_id, summary, gaps, server, tool_name

FetchResult:      chunks (list[CacheChunk]), total_chunks
```

**What to test**: Config loads from env, Redis connects, vLLM embedding returns vectors, Bedrock converse returns summary, FastMCP server starts on port.

**Context needed from this doc**: Sections 2, 3, 4 (stack, structure, config).

---

### Session 2: Tool Sync + Registry

**Goal**: LiteLLM registry client, sync job logic, Redis index creation, REST sync endpoints.

**Files to create**:
- `mcp_tool_manager/services/registry.py` — fetch servers, fetch tools, compute content_hash, diff logic, batch embed + store
- `mcp_tool_manager/api/sync.py` — POST /v1/sync, GET /v1/sync/{job_id}
- `mcp_tool_manager/api/health.py` — GET /v1/health

**Key decisions for implementation**:
- Use `redisvl.SearchIndex` with the YAML schema (Section 5.1) to create tool index
- Use `redisvl.SearchIndex` with cache schema (Section 5.3) to create cache index
- Index creation happens in `main.py` lifespan startup
- Sync job runs as `asyncio.create_task` (not a separate worker)
- Job status stored in Redis with `sync:job:{job_id}` key, TTL 3600s
- Lock via `sync:lock` key with `NX` + TTL 300s
- Embedding text composition follows Section 5.1 pattern
- Content hash: `hashlib.sha256(json.dumps(canonical, sort_keys=True)).hexdigest()[:16]`

**What to test**: Sync job fetches from LiteLLM, diffs correctly, creates/updates/deletes tools in Redis, job status polling works.

**Context needed from this doc**: Sections 5.1 (index schema, embedding text), 6 (sync endpoints).

---

### Session 3: find_tools

**Goal**: Hybrid search tool with @hint parsing (multi-field), query embedding cache, RRF fusion.

**Files to create**:
- `mcp_tool_manager/services/search.py` — parse @hints, build HybridQuery with multi-field matching, post-process
- `mcp_tool_manager/tools/find.py` — MCP tool registration with FastMCP

**Key decisions for implementation**:
- Use `redisvl.query.HybridQuery` with `combination_method="RRF"` (default)
- For Redis < 8.4: fallback to `AggregateHybridQuery` (same API)
- Query embedding cache: Redis key `qemb:{sha256(query)[:16]}`, store as bytes, TTL 300s
- `@hint` parsing: `re.findall(r'@(\S+)', query)` — extracts ALL hints, strips from clean_query
- Each hint matches against server_name (TAG), tags (TAG), AND is injected as boosted BM25 term for name/description substring matching
- Multiple hints combine with AND: `@github @api` = must relate to github AND api
- Post-processing: substring boost — for each hint, if it appears as substring in tool name, server_name, or tags, multiply score by 2x. This catches partial matches TAG filter misses (e.g., `@search` boosts `search_repos`)
- Return schema: list of ToolSearchResult with input_schema_summary (first 200 chars of schema description)

**Tool registration pattern** (conceptual):
```
# In tools/find.py
@mcp.tool()
async def find_tools(query: str, top_k: int = 10, ..., ctx: Context) -> list[dict]:
    """Discover available tools. Describe WHAT you need..."""
    await ctx.report_progress(0.1, 1.0, "Searching tools...")
    results = await search_service.hybrid_search(query, top_k)
    await ctx.report_progress(1.0, 1.0, f"Found {len(results)} tools")
    return [r.model_dump() for r in results]
```

**What to test**: Single @hint, multiple @hints, partial substring match, hybrid search ranked results, cache hit skips vLLM, min_score filtering.

**Context needed from this doc**: Sections 5.1 (full find_tools design).

---

### Session 4: call_tool

**Goal**: Tool execution with validation, MCP client pooling, async cache queue, LLM summary.

**Files to create**:
- `mcp_tool_manager/services/executor.py` — MCP ClientSession pool, call downstream
- `mcp_tool_manager/services/summariser.py` — boto3 Bedrock converse wrapper
- `mcp_tool_manager/services/cache.py` — async queue consumer, chunk + embed + store
- `mcp_tool_manager/tools/call.py` — MCP tool registration

**Key decisions for implementation**:
- **MCP Client Pool**: Dict of `server_name → ClientSession`. Create on first use. Use `mcp.ClientSession` with `StreamableHTTPTransport` or `SSETransport` based on config.
- **Server endpoint resolution**: First check Redis tool doc's server metadata. Fall back to LiteLLM registry API.
- **Validation**: `jsonschema.validate(arguments, tool_doc.input_schema)`. Raise clear error on failure.
- **Summary**: `boto3.client("bedrock-runtime", region_name=settings.bedrock.region).converse(modelId=settings.bedrock.summary_model, messages=[...])`. Keep prompt under 500 tokens. Extract summary + gaps. Use `asyncio.to_thread()` to avoid blocking (boto3 is sync).
- **Progress**: Report at 10% (connecting), 50% (executing), 80% (summarising), 100% (done).
- **Sampling pass-through**: If downstream server calls `create_message`, the executor's sampling_handler forwards to `ctx.session.create_message()`.
- **Elicitation pass-through**: Same pattern with `ctx.session.elicit()`.
- **Async cache**: `asyncio.Queue(maxsize=100)`. Background task started in lifespan. Chunks text, embeds via vLLM, stores with TTL.

**What to test**: Valid args pass, invalid args rejected, downstream MCP call works, summary generated, cache chunks appear in Redis.

**Context needed from this doc**: Sections 5.2 (full call_tool design).

---

### Session 5: fetch_cached_data

**Goal**: Session context retrieval with all three modes.

**Files to create**:
- `mcp_tool_manager/tools/fetch.py` — MCP tool registration

**Key decisions for implementation**:
- **By tool_call_id**: `FilterQuery` with `Tag("session_id") == sid & Tag("tool_call_id") == tcid`
- **By query**: `VectorQuery` with embedded query + `Tag("session_id") == sid` filter
- **All session**: `FilterQuery` with `Tag("session_id") == sid`, sorted by `created_at` desc
- Use `redisvl.query.filter` for all filter construction — no raw Redis commands

**What to test**: All three retrieval modes return correct chunks, semantic query ranks relevant chunks higher.

**Context needed from this doc**: Section 5.3.

---

### Session 6: Observability + Metrics + Polish

**Goal**: Full telemetry instrumentation, metrics endpoint, Docker compose, error handling.

**Files to create/modify**:
- `mcp_tool_manager/api/metrics.py` — GET /v1/metrics
- `docker-compose.yml` — redis, phoenix, app
- `Dockerfile` — multi-stage build
- Instrument all tools/ and services/ with spans and metrics

**Key decisions for implementation**:
- **Span instrumentation**: Use `tracer.start_as_current_span("find_tools")` wrapper. Set OpenInference attributes.
- **Phoenix custom keys**: `input.value`, `output.value`, `retrieval.query`, `retrieval.documents`, `llm.model_name`, `llm.token_count.prompt`, `llm.token_count.completion`, `session.id`, `tool.name`.
- **Metrics**: Use the registry pattern from Section 7. `meter.create_counter(...)`, `meter.create_histogram(...)`.
- **Error handling**: Catch `jsonschema.ValidationError`, `httpx.TimeoutException`, `redis.ConnectionError`. Return structured errors with `retryable` flag.
- **Graceful shutdown**: Drain async queue, close MCP sessions, close Redis pool.

**What to test**: Spans appear in Phoenix, metrics endpoint returns data, Docker compose starts all services.

**Context needed from this doc**: Section 7 (full telemetry design).

---

## 9. Redis Key Patterns

```
tool:{server_name}:{tool_name}           # Tool document (hash)
cache:{session_id}:{tool_call_id}:{idx}  # Cache chunk (hash, with TTL)
qemb:{query_hash}                        # Query embedding cache (binary, TTL 300s)
sync:lock                                # Sync job lock (string, TTL 300s)
sync:job:{job_id}                        # Job status (JSON string, TTL 3600s)
```

---

## 10. Latency Budget

| Operation | Target p50 | Target p95 | Bottleneck |
|---|---|---|---|
| find_tools (cache miss) | 60ms | 120ms | vLLM embed ~50ms |
| find_tools (cache hit) | 10ms | 25ms | Redis only |
| call_tool (excl. downstream) | 200ms | 500ms | LLM summary |
| call_tool (downstream MCP) | varies | varies | External server |
| fetch_cached_data | 5ms | 20ms | Redis only |
| sync (150 tools, 5 changed) | 400ms | 1s | vLLM batch embed |

---

## 11. A2A Extension Points (Future, Not in Scope)

The architecture naturally extends to Agent-to-Agent:

- **Agent Registry**: Same `redisvl` index pattern as tools. Schema adds `agent_card` (A2A spec), `capabilities`, `endpoint`.
- **find_agents**: Same hybrid search, different index.
- **call_agent**: Similar to `call_tool` but uses A2A protocol (HTTP JSON-RPC) instead of MCP.
- **Shared cache**: Same session cache serves both tool and agent results.
- **Config extension**: Add `A2ASettings` sub-model with `env_prefix="A2A__"`.

No code changes needed to existing modules — just new `tools/find_agent.py`, `tools/call_agent.py`, `services/a2a_client.py`.

---

## 12. Key Differences from Reference Design

| Aspect | Reference Design | This Design | Why |
|---|---|---|---|
| Embeddings per tool | 3 (multi-vector) | 1 (single combined) | Redis `FT.HYBRID` with BM25 field weighting handles disambiguation. 3x fewer embedding calls. |
| Vector search | Raw `FT.SEARCH` commands | `redisvl.HybridQuery` | Built-in RRF/LINEAR fusion. Zero manual score math. |
| Score fusion | Manual RRF implementation | Redis-native RRF | `FT.HYBRID` does it server-side. Faster, fewer lines. |
| Secrets | AWS Secrets Manager + cache | Env vars via pydantic-settings | Simpler. Use Vault/AWS at infra layer if needed. |
| SSL | Custom SSL context builder | httpx defaults + `SSL_CERT_FILE` env | httpx handles this natively. |
| MCP framework | fastmcp (old import path) | `fastmcp>=3.1` (PrefectHQ standalone) | Latest: auth, OTEL, Apps support built-in. |
| Telemetry | Raw OTEL SDK setup | `phoenix.otel.register()` | One-liner. Auto-discovers instrumentors. |
| Cache writes | Inline (blocking) | `asyncio.Queue` (non-blocking) | Same idea, stdlib only. |
| Summary | Not in reference | `boto3` Bedrock `converse` API | Agent needs concise context, not raw dumps. |
| Estimated file count | ~25 files | ~16 files | Fewer modules, each does more. |

---

## Appendix A: Dependencies (pyproject.toml)

```toml
[project]
name = "mcp-tool-manager"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "fastmcp>=3.1",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "redisvl>=0.16",
    "redis[hiredis]>=5.2",
    "httpx>=0.28",
    "pydantic>=2.10",
    "pydantic-settings>=2.7",
    "jsonschema>=4.23",
    "boto3>=1.35",
    "arize-phoenix-otel>=0.13",
    "openinference-semantic-conventions>=0.1",
    "opentelemetry-api>=1.30",
    "opentelemetry-sdk>=1.30",
    "opentelemetry-exporter-otlp>=1.30",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.9",
    "mypy>=1.14",
    "fakeredis>=2.26",
    "boto3-stubs[bedrock-runtime]>=1.35",
    "moto[bedrock]>=5.0",
]
```

## Appendix B: Docker Compose

```yaml
services:
  redis:
    image: redis/redis-stack:latest    # Redis 8.4+ with RediSearch
    ports: ["6379:6379", "8001:8001"]  # 8001 = RedisInsight

  phoenix:
    image: arizephoenix/phoenix:latest
    ports: ["6006:6006"]
    environment:
      PHOENIX_WORKING_DIR: /data

  app:
    build: .
    ports: ["8000:8000"]
    depends_on: [redis, phoenix]
    env_file: .env
```

## Appendix C: MCP Feature Checklist

| MCP Feature | Supported | Implementation |
|---|---|---|
| Tools (list + call) | ✅ | `@mcp.tool()` decorator via FastMCP |
| Resources | ✅ | Optional: expose tool index as resource |
| Prompts | ✅ | Optional: predefined search prompts |
| Progress notifications | ✅ | `ctx.report_progress()` in all tools |
| Sampling (server→client) | ✅ | Pass-through via `ctx.session.create_message()` |
| Elicitation | ✅ | Pass-through via `ctx.session.elicit()` |
| Logging | ✅ | `ctx.info()`, `ctx.debug()`, `ctx.error()` |
| Streamable HTTP transport | ✅ | FastMCP default |
| Stdio transport | ✅ | FastMCP `mcp.run(transport="stdio")` |
| Tool list changed notification | ✅ | `ctx.send_tool_list_changed()` after sync |
| Structured output (outputSchema) | ✅ | FastMCP auto-validates return types |
