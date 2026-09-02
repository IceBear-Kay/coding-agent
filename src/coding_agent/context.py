"""Deterministic software-level context byte budget calculations."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from coding_agent.models import Message

DEFAULT_MAX_CONTEXT_BYTES = 8_388_608
DEFAULT_MAX_CONTEXT_TOKENS = 524_288
DEFAULT_TOKEN_BYTES = 4
ContextPolicy = Literal["stop", "trim"]


class ContextSerializationError(ValueError):
    """Raised when the internal context cannot be represented as JSON."""


class ContextHistoryError(ValueError):
    """Raised when conversation history cannot be safely used as context."""

    def __init__(self) -> None:
        super().__init__("context history is invalid")


class ContextLimitError(ValueError):
    """Describe a context budget violation without retaining message contents."""

    def __init__(self, used_bytes: int, max_bytes: int) -> None:
        self.used_bytes = used_bytes
        self.max_bytes = max_bytes
        super().__init__(f"context input exceeds byte budget: {used_bytes} > {max_bytes}")


class ContextTokenLimitError(ValueError):
    """Describe an estimated input-token budget violation."""

    def __init__(self, used_tokens: int, max_tokens: int) -> None:
        self.used_tokens = used_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"estimated context input exceeds token budget: {used_tokens} > {max_tokens}"
        )


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


def estimate_context_tokens(
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, Any]],
) -> int:
    """Estimate input tokens from the serialized request representation.

    This provider-independent heuristic is not an exact model tokenizer. It is
    kept next to serialization so reasoning fields, tool arguments and schemas
    are included in the same stable input view.
    """
    serialized = serialize_context(messages, tools)
    return max(1, math.ceil(len(serialized) / DEFAULT_TOKEN_BYTES))


estimate_input_tokens = estimate_context_tokens


@dataclass(frozen=True, slots=True)
class ContextBudgetResult:
    """Immutable result of comparing measured context bytes with a budget."""

    used_bytes: int
    max_bytes: int
    used_tokens: int | None = None
    max_tokens: int | None = None

    @property
    def within_budget(self) -> bool:
        """Whether the measured bytes are allowed, including an exact-boundary match."""
        if self.used_bytes > self.max_bytes:
            return False
        return self.max_tokens is None or (
            self.used_tokens is not None and self.used_tokens <= self.max_tokens
        )

    @property
    def exceeded(self) -> bool:
        """Whether the measured bytes exceed the configured budget."""
        return not self.within_budget


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Configuration and pure operations for the software context byte budget."""

    max_bytes: int = DEFAULT_MAX_CONTEXT_BYTES
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int):
            raise TypeError("max_bytes must be an integer")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero")
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
                raise TypeError("max_tokens must be an integer")
            if self.max_tokens <= 0:
                raise ValueError("max_tokens must be greater than zero")

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
            used_tokens=(estimate_context_tokens(messages, tools) if self.max_tokens else None),
            max_tokens=self.max_tokens,
        )


@dataclass(frozen=True, slots=True)
class ContextSelectionResult:
    """A non-mutating request context selected from complete conversation history."""

    messages: tuple[Message, ...]
    used_bytes: int
    max_bytes: int
    removed_task_count: int = 0
    used_tokens: int = 0
    max_tokens: int | None = None

    @property
    def within_budget(self) -> bool:
        """Whether the selected request context fits the configured byte budget."""
        return self.used_bytes <= self.max_bytes and (
            self.max_tokens is None or self.used_tokens <= self.max_tokens
        )

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
    max_context_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
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
    if max_context_tokens is not None:
        if isinstance(max_context_tokens, bool) or not isinstance(max_context_tokens, int):
            raise TypeError("max_context_tokens must be an integer")
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than zero")
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
    used_tokens = estimate_context_tokens(selected, tools)
    removed_task_count = 0

    def exceeds() -> bool:
        return used_bytes > budget.max_bytes or (
            max_context_tokens is not None and used_tokens > max_context_tokens
        )

    if policy == "trim" and exceeds():
        for start, end in groups:
            removed_indices.update(
                index for index in range(start, end) if messages[index].role != "system"
            )
            removed_task_count += 1
            selected = [
                message for index, message in enumerate(messages) if index not in removed_indices
            ]
            used_bytes = budget.measure(selected, tools)
            used_tokens = estimate_context_tokens(selected, tools)
            if not exceeds():
                break

    return ContextSelectionResult(
        messages=tuple(message.model_copy(deep=True) for message in selected),
        used_bytes=used_bytes,
        max_bytes=budget.max_bytes,
        removed_task_count=removed_task_count,
        used_tokens=used_tokens,
        max_tokens=max_context_tokens,
    )


def _completed_task_ranges(
    messages: Sequence[Message], current_task_start: int
) -> list[tuple[int, int]]:
    """Validate history and find complete task ranges without splitting exchanges."""
    ranges: list[tuple[int, int]] = []
    task_start: int | None = None
    for index, message in enumerate(messages):
        if message.role == "system":
            if task_start is not None:
                raise ContextHistoryError
            continue
        if message.role == "user":
            if task_start is not None:
                _validate_task(messages, task_start, index, require_completion=True)
                ranges.append((task_start, index))
            task_start = index
            if index == current_task_start:
                break
        elif task_start is None:
            raise ContextHistoryError
    if task_start != current_task_start:
        raise ContextHistoryError
    _validate_task(messages, current_task_start, len(messages), require_completion=False)
    return ranges


def _validate_task(
    messages: Sequence[Message],
    start: int,
    end: int,
    *,
    require_completion: bool,
) -> None:
    """Validate one task's assistant/tool protocol without exposing its contents."""
    pending: dict[str, None] = {}
    seen_call_ids: set[str] = set()
    completed = False

    for message in messages[start + 1 : end]:
        if message.role == "system" or message.role == "user":
            raise ContextHistoryError
        if message.role == "assistant":
            if completed or pending:
                raise ContextHistoryError
            if message.tool_calls:
                call_ids = [tool_call.id for tool_call in message.tool_calls]
                if len(call_ids) != len(set(call_ids)) or seen_call_ids.intersection(call_ids):
                    raise ContextHistoryError
                seen_call_ids.update(call_ids)
                pending = dict.fromkeys(call_ids)
            elif message.content:
                completed = True
            else:
                raise ContextHistoryError
            continue
        if message.role == "tool":
            if message.tool_call_id not in pending:
                raise ContextHistoryError
            del pending[message.tool_call_id]
            continue
        raise ContextHistoryError

    if pending or (require_completion and not completed):
        raise ContextHistoryError


def check_context_budget(
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, Any]],
    max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
    max_context_tokens: int | None = None,
) -> ContextBudgetResult:
    """Measure a context and return a safe, non-mutating budget result."""
    return ContextBudget(max_bytes=max_context_bytes, max_tokens=max_context_tokens).check(
        messages, tools
    )


def validate_completed_history(messages: Sequence[Message]) -> None:
    """Validate that a history contains only complete, well-formed task exchanges."""
    if not messages:
        return

    task_start: int | None = None
    saw_user_task = False
    for index, message in enumerate(messages):
        if message.role == "system":
            if task_start is not None:
                raise ContextHistoryError
            continue
        if message.role == "user":
            saw_user_task = True
            if task_start is not None:
                _validate_task(messages, task_start, index, require_completion=True)
            task_start = index
            continue
        if task_start is None:
            raise ContextHistoryError

    if task_start is None or not saw_user_task:
        raise ContextHistoryError
    _validate_task(messages, task_start, len(messages), require_completion=True)


__all__ = [
    "DEFAULT_MAX_CONTEXT_BYTES",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "ContextBudget",
    "ContextBudgetResult",
    "ContextHistoryError",
    "ContextPolicy",
    "ContextSelectionResult",
    "ContextLimitError",
    "ContextTokenLimitError",
    "ContextSerializationError",
    "check_context_budget",
    "measure_context_bytes",
    "estimate_context_tokens",
    "estimate_input_tokens",
    "select_context",
    "serialize_context",
    "validate_completed_history",
]
