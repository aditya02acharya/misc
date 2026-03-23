"""Shared test fixtures."""
import pytest
import fakeredis.aioredis


@pytest.fixture
async def fakeredis_client():
    """In-memory async Redis client for testing."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()
