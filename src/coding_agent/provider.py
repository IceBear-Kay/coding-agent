"""Provider interface for model completions."""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

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
