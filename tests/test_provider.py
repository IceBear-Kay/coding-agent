from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from coding_agent.config import ProviderConfig
from coding_agent.errors import (
    ProviderAuthenticationError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
)
from coding_agent.models import Message, ModelResponse
from coding_agent.provider import (
    ModelProvider,
    build_chat_completion_payload,
    send_chat_completion_request,
)


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


def provider_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="test-secret-key",
        base_url="https://api.example.com/v1",
        model="test-model",
        timeout_seconds=30,
    )


def test_send_chat_completion_request_posts_payload_and_returns_json() -> None:
    request_log: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_log.append(request)
        return httpx.Response(200, json={"id": "response_1", "choices": []})

    payload = {"model": "test-model", "messages": [], "tools": []}
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = send_chat_completion_request(provider_config(), payload, client)

    assert response == {"id": "response_1", "choices": []}
    assert len(request_log) == 1
    request = request_log[0]
    assert str(request.url) == "https://api.example.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-secret-key"
    assert request.read() == b'{"model":"test-model","messages":[],"tools":[]}'


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, ProviderAuthenticationError),
        (403, ProviderAuthenticationError),
        (429, ProviderRateLimitError),
        (500, ProviderServerError),
        (503, ProviderServerError),
        (400, ProviderRequestError),
    ],
)
def test_send_chat_completion_request_maps_http_errors(
    status_code: int,
    error_type: type[Exception],
) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status_code))

    with httpx.Client(transport=transport) as client, pytest.raises(error_type) as exc_info:
        send_chat_completion_request(provider_config(), {}, client)

    assert "test-secret-key" not in str(exc_info.value)


def test_send_chat_completion_request_maps_network_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ProviderNetworkError, match="network request failed"),
    ):
        send_chat_completion_request(provider_config(), {}, client)


@pytest.mark.parametrize("body", ["not-json", ["unexpected"]])
def test_send_chat_completion_request_rejects_unexpected_success_body(body: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(200, text=body)
        return httpx.Response(200, json=body)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ProviderResponseError),
    ):
        send_chat_completion_request(provider_config(), {}, client)
