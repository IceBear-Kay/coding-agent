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


class AgentState(BaseModel):
    """Mutable state carried across steps of an agent run."""

    workspace_root: Path
    max_steps: int = Field(gt=0)
    messages: list[Message] = Field(default_factory=list)
    step_count: int = Field(default=0, ge=0)
    # Count historical task groups omitted from provider request contexts.
    context_trimmed_tasks: int = Field(default=0, ge=0)
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
