from collections.abc import Sequence
from typing import Any

from coding_agent.config import ProviderConfig
from coding_agent.models import Message, ModelResponse
from coding_agent.provider import ModelProvider, build_chat_completion_payload


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


def test_build_chat_completion_payload_serializes_messages_and_tools() -> None:
    config = ProviderConfig(
        api_key="test-secret-key",
        base_url="https://api.example.com/v1",
        model="test-model",
        timeout_seconds=30,
    )
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Read README.md"),
    ]
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object"},
            },
        }
    ]

    payload = build_chat_completion_payload(config, messages, tool_schemas)

    assert payload == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read README.md"},
        ],
        "tools": tool_schemas,
    }
    assert "test-secret-key" not in repr(payload)


def test_build_chat_completion_payload_accepts_no_tools() -> None:
    config = ProviderConfig(
        api_key="test-secret-key",
        base_url="https://api.example.com/v1",
        model="test-model",
        timeout_seconds=30,
    )

    payload = build_chat_completion_payload(
        config,
        [Message(role="user", content="Hello")],
        [],
    )

    assert payload["tools"] == []
