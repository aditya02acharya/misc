"""AWS Bedrock converse wrapper for tool result summarisation."""
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def summarise_tool_result(
    tool_name: str,
    arguments: dict,
    raw_result: str,
    bedrock_client: Any,
    settings: Any,
) -> tuple[str, str]:
    """
    Summarise a tool result using AWS Bedrock.
    Returns (summary, gaps) tuple.
    """
    # Truncate raw result to keep prompt short
    result_preview = raw_result[:2000] if len(raw_result) > 2000 else raw_result

    prompt = (
        f"Tool '{tool_name}' was called with arguments: {arguments}\n\n"
        f"Result:\n{result_preview}\n\n"
        "Provide a concise summary of this result (2-3 sentences). "
        "Then on a new line starting with 'Gaps:', identify any missing information or follow-up actions needed."
    )

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    def _call():
        return bedrock_client.converse(
            modelId=settings.bedrock.summary_model,
            messages=messages,
            inferenceConfig={"maxTokens": settings.bedrock.max_tokens},
        )

    try:
        response = await asyncio.to_thread(_call)
        content = response["output"]["message"]["content"][0]["text"]

        # Split on "Gaps:" marker
        if "Gaps:" in content:
            parts = content.split("Gaps:", 1)
            summary = parts[0].strip()
            gaps = parts[1].strip()
        else:
            summary = content.strip()
            gaps = "No gaps identified."

        return summary, gaps

    except Exception as exc:
        logger.warning("Bedrock summarisation failed: %s", exc)
        return f"Tool '{tool_name}' executed successfully.", "Summary unavailable."
