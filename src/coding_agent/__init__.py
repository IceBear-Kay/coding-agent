"""Core package for the coding agent."""

from coding_agent.config import ProviderConfig
from coding_agent.models import (
    AgentState,
    Message,
    ModelResponse,
    ToolCall,
    ToolResult,
    Usage,
)
from coding_agent.provider import FakeProvider, ModelProvider, OpenAICompatibleProvider

__version__ = "0.1.0"

__all__ = [
    "AgentState",
    "FakeProvider",
    "Message",
    "ModelProvider",
    "ModelResponse",
    "OpenAICompatibleProvider",
    "ProviderConfig",
    "ToolCall",
    "ToolResult",
    "Usage",
]
