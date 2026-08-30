"""Shared data models used by the coding agent."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """A basic text message in a model conversation."""

    role: MessageRole
    content: str = Field(min_length=1)


class AgentState(BaseModel):
    """Mutable state carried across steps of an agent run."""

    workspace_root: Path
    max_steps: int = Field(gt=0)
    messages: list[Message] = Field(default_factory=list)
    step_count: int = Field(default=0, ge=0)
    stop_reason: str | None = Field(default=None, min_length=1)


class ToolCall(BaseModel):
    """A tool invocation requested by a model."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


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
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    finish_reason: str | None = None
