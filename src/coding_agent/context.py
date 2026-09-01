"""Deterministic software-level context byte budget calculations."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from coding_agent.models import Message

DEFAULT_MAX_CONTEXT_BYTES = 262_144


class ContextSerializationError(ValueError):
    """Raised when the internal context cannot be represented as JSON."""


class ContextLimitError(ValueError):
    """Describe a context budget violation without retaining message contents."""

    def __init__(self, used_bytes: int, max_bytes: int) -> None:
        self.used_bytes = used_bytes
        self.max_bytes = max_bytes
        super().__init__(f"context input exceeds byte budget: {used_bytes} > {max_bytes}")


def serialize_context(
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, Any]],
) -> bytes:
    """Serialize messages and tool schemas using one stable compact JSON format."""
    try:
        payload = {
            "messages": [
                message.model_dump(mode="json", exclude_none=True) for message in messages
            ],
            "tools": list(tools),
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContextSerializationError("context serialization failed") from exc


def measure_context_bytes(
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, Any]],
) -> int:
    """Return the UTF-8 byte length of the normalized context representation."""
    return len(serialize_context(messages, tools))


@dataclass(frozen=True, slots=True)
class ContextBudgetResult:
    """Immutable result of comparing measured context bytes with a budget."""

    used_bytes: int
    max_bytes: int

    @property
    def within_budget(self) -> bool:
        """Whether the measured bytes are allowed, including an exact-boundary match."""
        return self.used_bytes <= self.max_bytes

    @property
    def exceeded(self) -> bool:
        """Whether the measured bytes exceed the configured budget."""
        return not self.within_budget


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Configuration and pure operations for the software context byte budget."""

    max_bytes: int = DEFAULT_MAX_CONTEXT_BYTES

    def __post_init__(self) -> None:
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")

    def measure(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]],
    ) -> int:
        """Measure a context without changing either input sequence."""
        return measure_context_bytes(messages, tools)

    def check(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]],
    ) -> ContextBudgetResult:
        """Measure a context and compare it against this budget."""
        return ContextBudgetResult(
            used_bytes=self.measure(messages, tools),
            max_bytes=self.max_bytes,
        )


def check_context_budget(
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, Any]],
    max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
) -> ContextBudgetResult:
    """Measure a context and return a safe, non-mutating budget result."""
    return ContextBudget(max_bytes=max_context_bytes).check(messages, tools)


__all__ = [
    "DEFAULT_MAX_CONTEXT_BYTES",
    "ContextBudget",
    "ContextBudgetResult",
    "ContextLimitError",
    "ContextSerializationError",
    "check_context_budget",
    "measure_context_bytes",
    "serialize_context",
]
