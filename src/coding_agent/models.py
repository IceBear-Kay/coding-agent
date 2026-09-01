"""Shared data models used by the coding agent."""

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

MessageRole = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A tool invocation requested by a model."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class Message(BaseModel):
    """A conversation message exchanged with a model provider."""

    role: MessageRole
    content: str | None
    reasoning_content: str | None = Field(default=None, exclude_if=lambda value: value is None)
    tool_calls: list[ToolCall] = Field(default_factory=list, exclude_if=lambda value: not value)
    tool_call_id: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_role_fields(self) -> Self:
        if self.role in {"system", "user"} and (self.content is None or not self.content):
            raise ValueError(f"{self.role} messages require non-empty content")
        if self.role == "tool":
            if self.tool_call_id is None:
                raise ValueError("tool messages require tool_call_id")
            if self.content is None:
                raise ValueError("tool messages require content")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("tool_calls are only valid for assistant messages")
        if self.role != "assistant" and self.reasoning_content is not None:
            raise ValueError("reasoning_content is only valid for assistant messages")
        return self


class TaskStats(BaseModel):
    """Non-authoritative diagnostics collected for one task run."""

    provider_attempts: int = Field(default=0, ge=0)
    tool_dispatches: int = Field(default=0, ge=0)
    tool_errors: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    known_usage_requests: int = Field(default=0, ge=0)
    unknown_usage_requests: int = Field(default=0, ge=0)
    context_bytes: int | None = Field(default=None, ge=0)
    context_max_bytes: int | None = Field(default=None, ge=1)
    context_trimmed_tasks: int = Field(default=0, ge=0)
    runtime_seconds: float = Field(default=0.0, ge=0.0)

    @property
    def usage_complete(self) -> bool:
        """Whether every provider attempt returned a usable usage block."""
        return self.provider_attempts > 0 and self.unknown_usage_requests == 0

    @property
    def known_input_tokens(self) -> int | None:
        return self.input_tokens if self.known_usage_requests else None

    @property
    def known_output_tokens(self) -> int | None:
        return self.output_tokens if self.known_usage_requests else None

    @property
    def known_total_tokens(self) -> int | None:
        return self.total_tokens if self.known_usage_requests else None

    @property
    def model_requests(self) -> int:
        """Compatibility alias for the number of provider attempts."""
        return self.provider_attempts

    @property
    def tool_calls(self) -> int:
        """Compatibility alias for dispatcher entries."""
        return self.tool_dispatches

    @property
    def missing_usage_requests(self) -> int:
        return self.unknown_usage_requests

    @property
    def duration_seconds(self) -> float:
        return self.runtime_seconds


class AgentState(BaseModel):
    """Mutable state carried across steps of an agent run."""

    workspace_root: Path
    max_steps: int = Field(gt=0)
    messages: list[Message] = Field(default_factory=list)
    step_count: int = Field(default=0, ge=0)
    # Count historical task groups omitted from provider request contexts.
    context_trimmed_tasks: int = Field(default=0, ge=0)
    stats: TaskStats = Field(default_factory=TaskStats)
    stop_reason: str | None = Field(default=None, min_length=1)


class SessionState(BaseModel):
    """In-memory history committed only after a task completes normally."""

    messages: list[Message] = Field(default_factory=list)


class ToolResult(BaseModel):
    """The output produced for a specific tool invocation."""

    tool_call_id: str = Field(min_length=1)
    content: str
    is_error: bool = False


class Usage(BaseModel):
    """Token counts reported for one model response."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    """Provider-independent output returned to the agent loop."""

    text: str | None = None
    reasoning_content: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    finish_reason: str | None = None
