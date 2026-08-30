"""Provider interface for model completions."""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from coding_agent.config import ProviderConfig
from coding_agent.models import Message, ModelResponse


@runtime_checkable
class ModelProvider(Protocol):
    """Return provider-independent responses for model conversations."""

    def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        """Generate the next model response for the current conversation."""
        ...


def build_chat_completion_payload(
    config: ProviderConfig,
    messages: Sequence[Message],
    tool_schemas: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build the JSON-compatible body for an OpenAI-compatible chat request."""
    return {
        "model": config.model,
        "messages": [message.model_dump() for message in messages],
        "tools": list(tool_schemas),
    }
