"""Provider interface for model completions."""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import httpx

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
            response = request_client.post(url, headers=headers, json=payload)
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
