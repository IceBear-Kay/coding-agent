"""Shared data models used by the coding agent."""

from typing import Literal

from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant"]


class Message(BaseModel):
    """A basic text message in a model conversation."""

    role: MessageRole
    content: str = Field(min_length=1)
