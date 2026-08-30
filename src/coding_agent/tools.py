"""Safe local tool definitions, registration, dispatch, and workspace paths."""

import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from coding_agent.models import ToolCall, ToolResult

ParametersT = TypeVar("ParametersT", bound=BaseModel)
ToolHandler = Callable[[ParametersT], Any]
DEFAULT_MAX_OUTPUT_CHARS = 32_000
DEFAULT_MAX_LIST_ENTRIES = 1_000
TRUNCATION_MARKER = "\n...[output truncated]"
LIST_TRUNCATION_MARKER = "\n...[file list truncated]"
IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
    }
)


class ToolError(Exception):
    """Base class for errors that can be returned as a tool result."""


class WorkspacePathError(ToolError, ValueError):
    """The requested path is invalid or outside the workspace."""


class WorkspaceFileError(ToolError):
    """The requested workspace file cannot be read as UTF-8 text."""


class ToolRegistrationError(ToolError, ValueError):
    """A tool cannot be registered in the current registry."""


class UnknownToolError(ToolError, LookupError):
    """A requested tool is not present in the registry."""


class Workspace:
    """Resolve paths while keeping all access inside one workspace root."""

    def __init__(self, root: str | Path) -> None:
        try:
            resolved_root = Path(root).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("Workspace root must be an existing directory") from exc

        if not resolved_root.is_dir():
            raise ValueError("Workspace root must be an existing directory")
        self.root = resolved_root

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a relative path and reject escapes, including symlinks."""
        if not isinstance(path, (str, Path)):
            raise WorkspacePathError("Workspace path must be a relative string")

        if isinstance(path, str) and not path:
            raise WorkspacePathError("Workspace path must not be empty")

        candidate = Path(path)
        windows_candidate = PureWindowsPath(str(path))
        if (
            candidate.is_absolute()
            or candidate.anchor
            or windows_candidate.is_absolute()
            or windows_candidate.anchor
        ):
            raise WorkspacePathError("Workspace path must be relative")

        try:
            resolved = (self.root / candidate).resolve(strict=False)
            resolved.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkspacePathError("Workspace path escapes the workspace root") from exc
        return resolved

    def list_files(
        self,
        path: str | Path = ".",
        *,
        max_entries: int = DEFAULT_MAX_LIST_ENTRIES,
    ) -> str:
        """Return workspace-relative file paths below a directory in stable order."""
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")

        directory = self.resolve_path(path)
        if not directory.exists():
            raise FileNotFoundError(f"Workspace directory does not exist: {path}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Workspace path is not a directory: {path}")

        file_paths: list[str] = []
        for current_root, directory_names, file_names in os.walk(directory, followlinks=False):
            directory_names[:] = sorted(
                name for name in directory_names if name not in IGNORED_DIRECTORY_NAMES
            )
            for file_name in sorted(file_names):
                file_path = Path(current_root) / file_name
                if file_path.is_symlink():
                    try:
                        self.resolve_path(file_path.relative_to(self.root))
                    except WorkspacePathError:
                        continue
                relative_path = file_path.relative_to(self.root).as_posix()
                file_paths.append(relative_path)

        file_paths = sorted(file_paths)
        result = "\n".join(file_paths[:max_entries])
        if len(file_paths) > max_entries:
            result += LIST_TRUNCATION_MARKER
        return result

    def read_file(
        self,
        path: str | Path,
        *,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> str:
        """Read one UTF-8 text file and mark output that exceeds the limit."""
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be greater than zero")

        file_path = self.resolve_path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Workspace file does not exist: {path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Workspace path is not a file: {path}")

        try:
            raw_content = file_path.read_bytes()
        except OSError as exc:
            raise WorkspaceFileError(f"Unable to read workspace file: {path}") from exc
        if b"\x00" in raw_content:
            raise WorkspaceFileError("Workspace file appears to be binary; UTF-8 text required")
        try:
            content = raw_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceFileError("Workspace file is not valid UTF-8 text") from exc
        if len(content) > max_output_chars:
            return content[:max_output_chars] + TRUNCATION_MARKER
        return content


class ListFilesArguments(BaseModel):
    """Arguments accepted by the list_files tool."""

    model_config = ConfigDict(extra="forbid")

    path: str = "."


class ReadFileArguments(BaseModel):
    """Arguments accepted by the read_file tool."""

    model_config = ConfigDict(extra="forbid")

    path: str


class ToolSpec[ParametersT]:
    """Describe one callable tool and the model used to validate its arguments."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: type[ParametersT],
        handler: ToolHandler[ParametersT],
    ) -> None:
        if not name:
            raise ValueError("Tool name must not be empty")
        if not description:
            raise ValueError("Tool description must not be empty")
        if not isinstance(parameters, type) or not issubclass(parameters, BaseModel):
            raise TypeError("Tool parameters must be a Pydantic model class")
        if not callable(handler):
            raise TypeError("Tool handler must be callable")

        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for the tool's argument object."""
        return self.parameters.model_json_schema()

    def to_provider_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    @property
    def schema(self) -> dict[str, Any]:
        """Expose the provider schema as a convenient read-only property."""
        return self.to_provider_schema()


class ToolRegistry:
    """Maintain a deterministic set of uniquely named tools."""

    def __init__(self, specs: list[ToolSpec[Any]] | None = None) -> None:
        self._specs: dict[str, ToolSpec[Any]] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec[Any]) -> None:
        if spec.name in self._specs:
            raise ToolRegistrationError(f"Tool already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec[Any]:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool: {name}") from exc

    def schemas(self) -> list[dict[str, Any]]:
        """Return provider schemas in registration order."""
        return [spec.to_provider_schema() for spec in self._specs.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[ToolSpec[Any]]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)


def read_only_tool_specs(
    workspace: Workspace,
) -> tuple[ToolSpec[ListFilesArguments], ToolSpec[ReadFileArguments]]:
    """Build the standard read-only workspace tool specifications."""
    return (
        ToolSpec(
            name="list_files",
            description="List files under a workspace directory.",
            parameters=ListFilesArguments,
            handler=lambda arguments: workspace.list_files(arguments.path),
        ),
        ToolSpec(
            name="read_file",
            description="Read a UTF-8 text file from the workspace.",
            parameters=ReadFileArguments,
            handler=lambda arguments: workspace.read_file(arguments.path),
        ),
    )


def create_read_only_registry(workspace: Workspace) -> ToolRegistry:
    """Create a registry containing list_files and read_file."""
    return ToolRegistry(list(read_only_tool_specs(workspace)))


class ToolDispatcher:
    """Validate and execute tool calls, returning structured results."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def dispatch(self, tool_call: ToolCall) -> ToolResult:
        """Execute one call without allowing tool failures to escape."""
        try:
            spec = self.registry.get(tool_call.name)
        except UnknownToolError as exc:
            return ToolResult(tool_call_id=tool_call.id, content=str(exc), is_error=True)

        try:
            self._reject_extra_fields(spec.parameters, tool_call.arguments)
            arguments = spec.parameters.model_validate(tool_call.arguments)
        except (ValidationError, ValueError, TypeError) as exc:
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Invalid arguments for {tool_call.name}: {exc}",
                is_error=True,
            )

        try:
            output = spec.handler(arguments)
        except Exception as exc:  # Tool failures are data returned to the agent loop.
            return ToolResult(
                tool_call_id=tool_call.id,
                content=f"Tool {tool_call.name} failed: {exc}",
                is_error=True,
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            content=self._format_output(output),
        )

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Alias for dispatch, suitable for callers that use execute terminology."""
        return self.dispatch(tool_call)

    @staticmethod
    def _reject_extra_fields(
        parameters: type[BaseModel],
        arguments: Mapping[str, Any],
    ) -> None:
        field_names = set(parameters.model_fields)
        aliases = {
            field.alias for field in parameters.model_fields.values() if field.alias is not None
        }
        extra = sorted(set(arguments) - field_names - aliases)
        if extra:
            names = ", ".join(extra)
            raise ValueError(f"unexpected field(s): {names}")

    @staticmethod
    def _format_output(output: Any) -> str:
        if isinstance(output, str):
            return output
        if output is None:
            return ""
        return str(output)
