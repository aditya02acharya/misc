"""Hybrid search service: parse @hints, build HybridQuery, post-process results."""
import logging
import re
from typing import Any

from mcp_tool_manager.models import ToolSearchResult
from mcp_tool_manager.services.embedding import embed

logger = logging.getLogger(__name__)


def parse_hints(query: str) -> tuple[list[str], str]:
    """
    Extract @hint tokens from query.
    Returns (hints, clean_query).
    E.g. "@github search repos" -> (["github"], "search repos")
    """
    hints = re.findall(r"@(\S+)", query)
    clean_query = re.sub(r"@\S+\s*", "", query).strip()
    return hints, clean_query


def _build_hint_filter(hints: list[str]) -> Any | None:
    """Build redisvl filter expression for @hint TAG matching."""
    if not hints:
        return None

    try:
        from redisvl.query.filter import Tag

        # Each hint can match server_name, name_tag, or tags (OR across fields)
        # Multiple hints combined with AND
        combined = None
        for hint in hints:
            hint_lower = hint.lower()
            field_filter = (
                (Tag("server_name") == hint_lower)
                | (Tag("name_tag") == hint_lower)
                | (Tag("tags") == hint_lower)
            )
            combined = field_filter if combined is None else (combined & field_filter)

        return combined
    except Exception as exc:
        logger.warning("Could not build hint filter: %s", exc)
        return None


def _apply_substring_boost(results: list[dict], hints: list[str]) -> list[dict]:
    """
    Post-process: multiply score by 2x for each hint that appears as substring
    in tool name, server_name, or tags.
    """
    if not hints:
        return results

    boosted = []
    for item in results:
        score = float(item.get("vector_distance", 0) or item.get("score", 0))
        name = (item.get("name") or "").lower()
        server = (item.get("server_name") or "").lower()
        tags = (item.get("tags") or "").lower()
        description = (item.get("description") or "").lower()

        for hint in hints:
            h = hint.lower()
            if h in name or h in server or h in tags or h in description:
                score *= 2.0

        item = dict(item)
        item["_boosted_score"] = score
        boosted.append(item)

    return boosted


def _input_schema_summary(schema_json: str) -> str:
    """Return first 200 chars of input schema as summary."""
    if not schema_json:
        return ""
    return schema_json[:200]


async def hybrid_search(
    query: str,
    top_k: int,
    settings: Any,
    tool_index: Any,
    http_client: Any,
    redis_client: Any = None,
    server_filter: str | None = None,
) -> list[ToolSearchResult]:
    """
    Full hybrid search with @hint parsing, embedding, RRF fusion, and post-boost.
    Falls back gracefully if index is unavailable.
    """
    hints, clean_query = parse_hints(query)

    # If server_filter provided, add as hint
    if server_filter and server_filter not in hints:
        hints.append(server_filter)

    if not clean_query:
        clean_query = " ".join(hints)

    # Embed the clean query
    query_vector = await embed(clean_query, settings, http_client, redis_client)

    if tool_index is None:
        logger.warning("Tool index not available, returning empty results")
        return []

    # Build hint filter for TAG matching
    hint_filter = _build_hint_filter(hints)

    try:
        results = await _execute_hybrid_query(
            clean_query=clean_query,
            query_vector=query_vector,
            hints=hints,
            hint_filter=hint_filter,
            top_k=top_k * 2,  # over-fetch for post-processing
            settings=settings,
            tool_index=tool_index,
        )
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        return []

    # Post-process: substring boost
    results = _apply_substring_boost(results, hints)

    # Sort by boosted score descending
    results.sort(key=lambda x: x.get("_boosted_score", 0), reverse=True)

    # Apply min_score threshold and truncate
    min_score = settings.search.min_score
    final = []
    for item in results:
        if len(final) >= top_k:
            break
        score = item.get("_boosted_score", item.get("score", 0))
        if score < min_score:
            continue
        final.append(
            ToolSearchResult(
                tool_id=item.get("tool_id", ""),
                name=item.get("name", ""),
                server_name=item.get("server_name", ""),
                description=item.get("description", ""),
                input_schema_summary=_input_schema_summary(item.get("input_schema", "")),
                score=round(score, 4),
            )
        )

    return final


async def _execute_hybrid_query(
    clean_query: str,
    query_vector: bytes,
    hints: list[str],
    hint_filter: Any,
    top_k: int,
    settings: Any,
    tool_index: Any,
) -> list[dict]:
    """Execute hybrid (BM25 + vector) query via redisvl."""
    try:
        from redisvl.query import HybridQuery

        # Inject hints into BM25 text query for substring matching
        bm25_query = clean_query
        if hints:
            boosted_terms = " ".join(f"({h})" for h in hints)
            bm25_query = f"{clean_query} {boosted_terms}".strip()

        # Build combination kwargs based on method
        combo_kwargs = {}
        combination = getattr(settings.search, "combination", "RRF")
        combo_kwargs["combination_method"] = combination
        if combination == "LINEAR":
            combo_kwargs["text_weight"] = settings.search.text_weight
            combo_kwargs["vector_weight"] = settings.search.vector_weight

        hybrid_q = HybridQuery(
            text=bm25_query,
            text_field_name="searchable_text",
            vector=query_vector,
            vector_field_name="embedding",
            return_fields=[
                "tool_id",
                "name",
                "description",
                "server_name",
                "tags",
                "input_schema",
                "score",
            ],
            num_results=top_k,
            **combo_kwargs,
        )

        if hint_filter is not None:
            hybrid_q.set_filter(hint_filter)

        results = await tool_index.query(hybrid_q)
        return results if results else []

    except ImportError:
        logger.warning("redisvl HybridQuery not available, falling back to vector-only search")
        return await _vector_only_search(query_vector, top_k, settings, tool_index)
    except Exception as exc:
        logger.warning("HybridQuery failed (%s), falling back to vector-only search", exc)
        return await _vector_only_search(query_vector, top_k, settings, tool_index)


async def _vector_only_search(
    query_vector: bytes,
    top_k: int,
    settings: Any,
    tool_index: Any,
) -> list[dict]:
    """Vector-only fallback search."""
    from redisvl.query import VectorQuery

    vq = VectorQuery(
        vector=query_vector,
        vector_field_name="embedding",
        return_fields=["tool_id", "name", "description", "server_name", "tags", "input_schema"],
        num_results=top_k,
    )
    results = await tool_index.query(vq)
    return results if results else []
