"""Provider interface for model completions."""

import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import ValidationError

from coding_agent.config import ProviderConfig
from coding_agent.errors import (
    ProviderAuthenticationError,
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderServerError,
)
from coding_agent.models import Message, ModelResponse, ToolCall, Usage

EMPTY_CONTENT_FINISH_REASONS = frozenset(
    {
        "length",
        "content_filter",
        "insufficient_system_resource",
    }
)


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
        "messages": [serialize_message_for_api(message) for message in messages],
        "tools": list(tool_schemas),
    }


def serialize_message_for_api(message: Message) -> dict[str, Any]:
    """Convert one internal message into the provider's message format."""
    serialized: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }

    if message.reasoning_content is not None:
        serialized["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        serialized["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": _serialize_tool_arguments(tool_call.arguments),
                },
            }
            for tool_call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        serialized["tool_call_id"] = message.tool_call_id
    return serialized


def _serialize_tool_arguments(arguments: dict[str, Any]) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ProviderResponseError("Tool call arguments are not JSON serializable") from exc


def send_chat_completion_request(
    config: ProviderConfig,
    payload: dict[str, Any],
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Send a chat request and return its raw JSON object."""
    url = f"{str(config.base_url).rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key.get_secret_value()}",
        "Content-Type": "application/json",
    }
    owns_client = client is None
    request_client = client or httpx.Client(timeout=config.timeout_seconds)

    try:
        try:
            response = request_client.post(
                url,
                headers=headers,
                json=payload,
                timeout=config.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderNetworkError("Provider request timed out") from exc
        except httpx.NetworkError as exc:
            raise ProviderNetworkError("Provider network request failed") from exc
        except httpx.HTTPError as exc:
            raise ProviderNetworkError("Provider HTTP request failed") from exc

        if response.status_code in {401, 403}:
            raise ProviderAuthenticationError("Provider authentication failed")
        if response.status_code == 429:
            raise ProviderRateLimitError("Provider rate limit exceeded")
        if response.status_code >= 500:
            raise ProviderServerError("Provider server error")
        if response.status_code >= 400:
            raise ProviderRequestError(
                f"Provider request failed with status {response.status_code}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError("Provider returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise ProviderResponseError("Provider returned a non-object JSON response")
        return data
    finally:
        if owns_client:
            request_client.close()


def parse_chat_completion_response(raw_response: Any) -> ModelResponse:
    """Convert an OpenAI-compatible response object into a ModelResponse."""
    if not isinstance(raw_response, dict):
        raise ProviderResponseError("Provider response must be a JSON object")

    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseError("Provider response must contain at least one choice")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderResponseError("Provider choice must be a JSON object")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise ProviderResponseError("Provider choice is missing a message object")

    if message.get("role") != "assistant":
        raise ProviderResponseError("Provider message role must be assistant")
    if "content" not in message:
        raise ProviderResponseError("Provider message is missing content")

    text = message.get("content")
    if text is not None and not isinstance(text, str):
        raise ProviderResponseError("Provider message content must be a string or null")

    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ProviderResponseError("Provider finish_reason must be a string or null")

    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ProviderResponseError("Provider reasoning_content must be a string or null")

    raw_tool_calls = message.get("tool_calls", [])
    if not isinstance(raw_tool_calls, list):
        raise ProviderResponseError("Provider tool_calls must be a list")
    if not text and not raw_tool_calls and finish_reason not in EMPTY_CONTENT_FINISH_REASONS:
        raise ProviderResponseError("Provider message must contain content or tool_calls")

    tool_calls: list[ToolCall] = []
    for index, raw_tool_call in enumerate(raw_tool_calls):
        if not isinstance(raw_tool_call, dict):
            raise ProviderResponseError(f"Provider tool call {index} must be a JSON object")

        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            raise ProviderResponseError(f"Provider tool call {index} is missing a function object")

        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ProviderResponseError(
                f"Provider tool call {index} arguments must be a JSON string"
            )

        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"Provider tool call {index} has invalid arguments JSON"
            ) from exc

        if not isinstance(parsed_arguments, dict):
            raise ProviderResponseError(
                f"Provider tool call {index} arguments must decode to an object"
            )

        try:
            tool_calls.append(
                ToolCall(
                    id=raw_tool_call.get("id"),
                    name=function.get("name"),
                    arguments=parsed_arguments,
                )
            )
        except ValidationError as exc:
            raise ProviderResponseError(f"Provider tool call {index} has invalid fields") from exc

    usage = None
    raw_usage = raw_response.get("usage")
    if raw_usage is not None:
        if not isinstance(raw_usage, dict):
            raise ProviderResponseError("Provider usage must be a JSON object")
        try:
            usage = Usage.model_validate(
                {
                    "input_tokens": raw_usage["prompt_tokens"],
                    "output_tokens": raw_usage["completion_tokens"],
                    "total_tokens": raw_usage["total_tokens"],
                }
            )
        except (KeyError, ValidationError) as exc:
            raise ProviderResponseError("Provider usage has invalid fields") from exc

    return ModelResponse(
        text=text,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish_reason,
    )


class OpenAICompatibleProvider:
    """Provider implementation for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        config: ProviderConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.client = client

    def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        """Build, send, and parse one chat completion request."""
        payload = build_chat_completion_payload(self.config, messages, tool_schemas)
        raw_response = send_chat_completion_request(self.config, payload, self.client)
        return parse_chat_completion_response(raw_response)


class FakeProvider:
    """Deterministic provider test double with request history."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = [response.model_copy(deep=True) for response in responses]
        self.requests: list[tuple[list[Message], list[dict[str, Any]]]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        """Record one request and return the next configured response."""
        if not self._responses:
            raise ProviderResponseError("FakeProvider has no remaining responses")

        self.requests.append(
            (
                [message.model_copy(deep=True) for message in messages],
                deepcopy(list(tool_schemas)),
            )
        )
        return self._responses.pop(0).model_copy(deep=True)
