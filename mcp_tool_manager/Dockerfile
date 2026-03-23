FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies into /app/.venv
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY mcp_tool_manager/ ./mcp_tool_manager/

# Install the project itself
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the built venv and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/mcp_tool_manager /app/mcp_tool_manager

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "mcp_tool_manager.main"]
