"""Deterministic software-level context byte budget calculations."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from coding_agent.models import Message

DEFAULT_MAX_CONTEXT_BYTES = 262_144
ContextPolicy = Literal["stop", "trim"]


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


@dataclass(frozen=True, slots=True)
class ContextSelectionResult:
    """A non-mutating request context selected from complete conversation history."""

    messages: tuple[Message, ...]
    used_bytes: int
    max_bytes: int
    removed_task_count: int = 0

    @property
    def within_budget(self) -> bool:
        """Whether the selected request context fits the configured byte budget."""
        return self.used_bytes <= self.max_bytes

    @property
    def trimmed(self) -> bool:
        """Whether one or more completed historical tasks were removed."""
        return self.removed_task_count > 0


def select_context(
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, Any]],
    *,
    current_task_start: int,
    max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
    policy: ContextPolicy = "stop",
) -> ContextSelectionResult:
    """Select a request context while preserving complete task boundaries.

    ``current_task_start`` points to the user message that began the active task.
    In ``trim`` mode, completed tasks before that message are removed oldest-first
    until the serialized context fits or no removable task remains. The input
    messages and nested tool-call arguments are never mutated.
    """
    if policy not in {"stop", "trim"}:
        raise ValueError("context policy must be 'stop' or 'trim'")
    budget = ContextBudget(max_bytes=max_context_bytes)
    if isinstance(current_task_start, bool) or not isinstance(current_task_start, int):
        raise TypeError("current_task_start must be an integer")
    if current_task_start < 0 or current_task_start >= len(messages):
        raise ValueError("current_task_start must point to a message")
    if messages[current_task_start].role != "user":
        raise ValueError("current_task_start must point to a user message")

    groups = _completed_task_ranges(messages, current_task_start)
    removed_indices: set[int] = set()
    selected = list(messages)
    used_bytes = budget.measure(selected, tools)
    removed_task_count = 0
    if policy == "trim" and used_bytes > budget.max_bytes:
        for start, end in groups:
            removed_indices.update(
                index for index in range(start, end) if messages[index].role != "system"
            )
            removed_task_count += 1
            selected = [
                message for index, message in enumerate(messages) if index not in removed_indices
            ]
            used_bytes = budget.measure(selected, tools)
            if used_bytes <= budget.max_bytes:
                break

    return ContextSelectionResult(
        messages=tuple(message.model_copy(deep=True) for message in selected),
        used_bytes=used_bytes,
        max_bytes=budget.max_bytes,
        removed_task_count=removed_task_count,
    )


def _completed_task_ranges(
    messages: Sequence[Message], current_task_start: int
) -> list[tuple[int, int]]:
    """Find complete historical task ranges without splitting tool exchanges."""
    ranges: list[tuple[int, int]] = []
    task_start: int | None = None
    for index, message in enumerate(messages[:current_task_start]):
        if message.role == "system":
            continue
        if message.role == "user":
            if task_start is not None:
                ranges.append((task_start, index))
            task_start = index
        elif task_start is None:
            raise ValueError("history contains a message before its user task")
    if task_start is not None:
        ranges.append((task_start, current_task_start))
    return ranges


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
    "ContextPolicy",
    "ContextSelectionResult",
    "ContextLimitError",
    "ContextSerializationError",
    "check_context_budget",
    "measure_context_bytes",
    "select_context",
    "serialize_context",
]
