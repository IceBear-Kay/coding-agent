import json
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
from coding_agent.models import Message, ModelResponse, ToolCall
from coding_agent.provider import (
    FakeProvider,
    ModelProvider,
    OpenAICompatibleProvider,
    build_chat_completion_payload,
    parse_chat_completion_response,
    send_chat_completion_request,
    serialize_message_for_api,
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


def test_build_chat_completion_payload_includes_output_budget_when_requested() -> None:
    payload = build_chat_completion_payload(
        provider_config(),
        [Message(role="user", content="Hello")],
        [],
        max_tokens=123,
    )

    assert payload["max_tokens"] == 123


def test_serialize_message_for_api_preserves_reasoning_and_tool_call_fields() -> None:
    message = Message(
        role="assistant",
        content=None,
        reasoning_content="I need to inspect the file.",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="read_file",
                arguments={"path": "README.md"},
            )
        ],
    )

    serialized = serialize_message_for_api(message)

    assert serialized == {
        "role": "assistant",
        "content": None,
        "reasoning_content": "I need to inspect the file.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                },
            }
        ],
    }


def test_serialize_message_for_api_preserves_tool_result_message() -> None:
    message = Message(
        role="tool",
        content="README contents",
        tool_call_id="call_1",
    )

    assert serialize_message_for_api(message) == {
        "role": "tool",
        "content": "README contents",
        "tool_call_id": "call_1",
    }


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


def test_send_chat_completion_request_applies_configured_timeout_to_custom_client() -> None:
    observed_timeouts: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"choices": []})

    config = provider_config().model_copy(update={"timeout_seconds": 17})
    with httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=1,
    ) as client:
        send_chat_completion_request(config, {}, client)

    assert observed_timeouts == [{"connect": 17.0, "read": 17.0, "write": 17.0, "pool": 17.0}]


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


def test_parse_chat_completion_response_reads_text_usage_and_finish_reason() -> None:
    response = parse_chat_completion_response(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
            },
        }
    )

    assert response.text == "Done"
    assert response.tool_calls == []
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.model_dump() == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }


def test_parse_chat_completion_response_decodes_tool_call_arguments() -> None:
    response = parse_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "I should inspect the file.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    assert response.text is None
    assert response.reasoning_content == "I should inspect the file."
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}


def test_parse_chat_completion_response_preserves_multiple_tool_calls() -> None:
    raw_tool_calls = [
        {
            "id": "call_1",
            "function": {"name": "list_files", "arguments": '{"path":"."}'},
        },
        {
            "id": "call_2",
            "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
        },
    ]

    response = parse_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": raw_tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )

    assert [call.id for call in response.tool_calls] == ["call_1", "call_2"]


@pytest.mark.parametrize(
    "raw_response",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
    ],
)
def test_parse_chat_completion_response_rejects_missing_required_structure(
    raw_response: dict[str, Any],
) -> None:
    with pytest.raises(ProviderResponseError):
        parse_chat_completion_response(raw_response)


def test_parse_chat_completion_response_rejects_empty_message() -> None:
    with pytest.raises(ProviderResponseError):
        parse_chat_completion_response({"choices": [{"message": {}}]})


def test_parse_chat_completion_response_accepts_null_content_with_tool_call() -> None:
    response = parse_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )

    assert response.text is None
    assert response.tool_calls[0].name == "read_file"


@pytest.mark.parametrize(
    ("content", "finish_reason"),
    [(content, finish_reason) for content in (None, "") for finish_reason in (None, "stop")],
)
def test_parse_chat_completion_response_rejects_empty_completion_without_tool_call(
    content: str | None,
    finish_reason: str | None,
) -> None:
    with pytest.raises(ProviderResponseError):
        parse_chat_completion_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": finish_reason,
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    "finish_reason",
    [
        "length",
        "content_filter",
        "insufficient_system_resource",
    ],
)
@pytest.mark.parametrize("content", [None, ""])
def test_parse_chat_completion_response_accepts_empty_content_with_terminal_reason(
    content: str | None,
    finish_reason: str,
) -> None:
    response = parse_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "Partial reasoning",
                    },
                    "finish_reason": finish_reason,
                }
            ]
        }
    )

    assert response.text == content
    assert response.reasoning_content == "Partial reasoning"
    assert response.tool_calls == []
    assert response.finish_reason == finish_reason


@pytest.mark.parametrize(
    "arguments",
    ["not-json", "[]", "null"],
)
def test_parse_chat_completion_response_rejects_invalid_tool_arguments(arguments: str) -> None:
    raw_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "read_file", "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }

    with pytest.raises(ProviderResponseError):
        parse_chat_completion_response(raw_response)


def test_parse_chat_completion_response_rejects_invalid_usage() -> None:
    raw_response = {
        "choices": [{"message": {"role": "assistant", "content": "Done"}}],
        "usage": {"prompt_tokens": -1, "completion_tokens": 3, "total_tokens": 2},
    }

    with pytest.raises(ProviderResponseError):
        parse_chat_completion_response(raw_response)


def test_openai_compatible_provider_composes_request_and_parser() -> None:
    request_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Done"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(provider_config(), client)
        response = provider.complete(
            [Message(role="user", content="Inspect the project")],
            [],
            max_tokens=123,
        )

    assert isinstance(provider, ModelProvider)
    assert response == ModelResponse(text="Done", finish_reason="stop")
    assert request_bodies == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Inspect the project"}],
            "tools": [],
            "max_tokens": 123,
        }
    ]


def test_tool_call_response_round_trips_into_next_request() -> None:
    raw_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I should read the README.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    assistant_response = parse_chat_completion_response(raw_response)
    history = [
        Message(role="user", content="Read README.md"),
        Message(
            role="assistant",
            content=assistant_response.text,
            reasoning_content=assistant_response.reasoning_content,
            tool_calls=assistant_response.tool_calls,
        ),
        Message(
            role="tool",
            tool_call_id="call_1",
            content="README contents",
        ),
    ]

    payload = build_chat_completion_payload(provider_config(), history, [])

    assert payload["messages"] == [
        {"role": "user", "content": "Read README.md"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I should read the README.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "README contents",
            "tool_call_id": "call_1",
        },
    ]


def test_fake_provider_returns_responses_in_order_and_records_requests() -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="First", finish_reason="stop"),
            ModelResponse(text="Second", finish_reason="stop"),
        ]
    )
    first_messages = [Message(role="user", content="First task")]
    second_messages = [Message(role="user", content="Second task")]
    tools = [{"type": "function", "function": {"name": "list_files"}}]

    first_response = provider.complete(first_messages, tools)
    second_response = provider.complete(second_messages, [])

    assert isinstance(provider, ModelProvider)
    assert first_response.text == "First"
    assert second_response.text == "Second"
    assert provider.requests == [
        (first_messages, tools),
        (second_messages, []),
    ]


def test_fake_provider_does_not_share_request_or_response_mutable_data() -> None:
    response = ModelResponse(
        text="Inspect",
        tool_calls=[
            {
                "id": "call_1",
                "name": "read_file",
                "arguments": {"path": "README.md"},
            }
        ],
    )
    provider = FakeProvider([response])
    messages = [Message(role="user", content="Inspect")]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    returned = provider.complete(messages, tools)
    messages[0].content = "Changed"
    tools[0]["function"]["name"] = "changed"
    returned.tool_calls[0].arguments["path"] = "changed"

    assert provider.requests[0][0][0].content == "Inspect"
    assert provider.requests[0][1][0]["function"]["name"] == "read_file"
    assert response.tool_calls[0].arguments["path"] == "README.md"


def test_fake_provider_rejects_requests_after_responses_are_exhausted() -> None:
    provider = FakeProvider([ModelResponse(text="Done")])
    provider.complete([], [])

    with pytest.raises(ProviderResponseError, match="no remaining responses"):
        provider.complete([], [])
