from collections.abc import Sequence
from typing import Any

from coding_agent.models import Message, ModelResponse
from coding_agent.provider import ModelProvider


class RecordingProvider:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.messages: list[Message] = []
        self.tool_schemas: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        self.messages = list(messages)
        self.tool_schemas = list(tool_schemas)
        return self.response


def test_model_provider_accepts_a_structural_test_double() -> None:
    expected = ModelResponse(text="Done", finish_reason="stop")
    provider = RecordingProvider(expected)
    messages = [Message(role="user", content="Inspect the project")]
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object"},
            },
        }
    ]

    response = provider.complete(messages, tool_schemas)

    assert isinstance(provider, ModelProvider)
    assert provider.messages == messages
    assert provider.tool_schemas == tool_schemas
    assert response is expected


def test_model_provider_rejects_an_object_without_complete() -> None:
    assert not isinstance(object(), ModelProvider)
