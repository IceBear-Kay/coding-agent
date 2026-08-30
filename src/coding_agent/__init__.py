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
from coding_agent.tools import (
    DEFAULT_MAX_LIST_ENTRIES,
    DEFAULT_MAX_OUTPUT_CHARS,
    LIST_TRUNCATION_MARKER,
    TRUNCATION_MARKER,
    ListFilesArguments,
    ReadFileArguments,
    ToolDispatcher,
    ToolRegistrationError,
    ToolRegistry,
    ToolSpec,
    UnknownToolError,
    Workspace,
    WorkspaceFileError,
    WorkspacePathError,
    create_read_only_registry,
    read_only_tool_specs,
)

__version__ = "0.1.0"

__all__ = [
    "AgentState",
    "DEFAULT_MAX_LIST_ENTRIES",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "FakeProvider",
    "ListFilesArguments",
    "LIST_TRUNCATION_MARKER",
    "Message",
    "ModelProvider",
    "ModelResponse",
    "OpenAICompatibleProvider",
    "ProviderConfig",
    "ReadFileArguments",
    "ToolCall",
    "ToolDispatcher",
    "ToolRegistrationError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "TRUNCATION_MARKER",
    "Usage",
    "UnknownToolError",
    "Workspace",
    "WorkspaceFileError",
    "WorkspacePathError",
    "create_read_only_registry",
    "read_only_tool_specs",
]
