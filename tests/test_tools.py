import os
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from coding_agent.models import ToolCall, ToolResult
from coding_agent.tools import (
    ListFilesArguments,
    ReadFileArguments,
    ToolDispatcher,
    ToolRegistrationError,
    ToolRegistry,
    ToolSpec,
    Workspace,
    WorkspaceFileError,
    WorkspacePathError,
    create_read_only_registry,
)


class EchoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


def make_echo_spec() -> ToolSpec[EchoArguments]:
    return ToolSpec(
        name="echo",
        description="Return the supplied text.",
        parameters=EchoArguments,
        handler=lambda arguments: arguments.text,
    )


def test_tool_spec_exposes_provider_schema() -> None:
    schema = make_echo_spec().to_provider_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["description"] == "Return the supplied text."
    assert schema["function"]["parameters"]["properties"]["text"]["type"] == "string"
    assert schema["function"]["parameters"]["required"] == ["text"]


def test_registry_preserves_order_and_rejects_duplicates() -> None:
    registry = ToolRegistry()
    registry.register(make_echo_spec())

    assert len(registry) == 1
    assert registry.get("echo").name == "echo"
    assert registry.schemas()[0]["function"]["name"] == "echo"

    with pytest.raises(ToolRegistrationError, match="already registered"):
        registry.register(make_echo_spec())


def test_dispatcher_validates_and_executes_registered_tool() -> None:
    dispatcher = ToolDispatcher(ToolRegistry([make_echo_spec()]))
    result = dispatcher.dispatch(ToolCall(id="call_1", name="echo", arguments={"text": "hello"}))

    assert result.tool_call_id == "call_1"
    assert result.content == "hello"
    assert result.is_error is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "text"),
        ({"text": 42}, "text"),
        ({"text": "hello", "extra": True}, "unexpected"),
    ],
)
def test_dispatcher_returns_structured_argument_errors(
    arguments: dict[str, object],
    message: str,
) -> None:
    dispatcher = ToolDispatcher(ToolRegistry([make_echo_spec()]))

    result = dispatcher.dispatch(ToolCall(id="call_1", name="echo", arguments=arguments))

    assert result.is_error is True
    assert result.tool_call_id == "call_1"
    assert message in result.content


def test_dispatcher_returns_structured_unknown_tool_error() -> None:
    result = ToolDispatcher(ToolRegistry()).execute(
        ToolCall(id="call_1", name="missing", arguments={})
    )

    assert result.is_error is True
    assert result.tool_call_id == "call_1"
    assert "Unknown tool" in result.content


def test_dispatcher_converts_handler_failure_to_tool_result() -> None:
    spec = ToolSpec(
        name="broken",
        description="Always fails.",
        parameters=EchoArguments,
        handler=lambda _: 1 / 0,
    )
    result = ToolDispatcher(ToolRegistry([spec])).dispatch(
        ToolCall(id="call_1", name="broken", arguments={"text": "hello"})
    )

    assert result.is_error is True
    assert "Tool broken failed" in result.content


def test_workspace_resolves_paths_inside_root(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    assert workspace.resolve_path("docs/guide.md") == tmp_path / "docs" / "guide.md"
    assert workspace.resolve_path(".") == tmp_path


def test_list_files_returns_sorted_workspace_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "nested.txt").write_text("nested", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "ignored.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.txt").write_text("ignored", encoding="utf-8")

    result = Workspace(tmp_path).list_files()

    assert result.splitlines() == ["a/nested.txt", "b.txt"]


def test_list_files_marks_output_when_entry_limit_is_reached(tmp_path: Path) -> None:
    for name in ["c.txt", "a.txt", "b.txt"]:
        (tmp_path / name).write_text(name, encoding="utf-8")

    result = Workspace(tmp_path).list_files(max_entries=2)

    assert result == "a.txt\nb.txt\n...[file list truncated]"


def test_list_files_stops_descending_after_budget_is_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "a"
    first.mkdir()
    for name in ["one.txt", "two.txt", "three.txt"]:
        (first / name).write_text(name, encoding="utf-8")
    later = tmp_path / "z"
    later.mkdir()
    (later / "later.txt").write_text("later", encoding="utf-8")

    original_scandir = os.scandir
    scanned_directories: list[str] = []
    scanned_entries = 0

    class TrackingScanner:
        def __init__(self, path: str | os.PathLike[str]):
            self.scanner = original_scandir(path)

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal scanned_entries
            entry = next(self.scanner)
            scanned_entries += 1
            return entry

        def close(self):
            self.scanner.close()

    def tracking_scandir(path: str | os.PathLike[str]):
        scanned_directories.append(Path(path).relative_to(tmp_path).as_posix() or ".")
        return TrackingScanner(path)

    monkeypatch.setattr(os, "scandir", tracking_scandir)
    result = Workspace(tmp_path).list_files(max_entries=1)

    assert result == "a/one.txt\n...[file list truncated]"
    assert scanned_directories == [".", "a"]
    assert scanned_entries == 3


def test_list_files_skips_directory_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    assert "secret.txt" not in Workspace(tmp_path).list_files()


def test_list_files_skips_reparse_directory_before_descending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked = tmp_path / "junction"
    linked.mkdir()
    (linked / "should-not-list.txt").write_text("hidden", encoding="utf-8")

    monkeypatch.setattr(
        Workspace,
        "_is_reparse_point",
        staticmethod(lambda path: path.name == "junction"),
    )

    assert "should-not-list.txt" not in Workspace(tmp_path).list_files()


def test_list_files_permission_failure_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied_scandir(_: str | os.PathLike[str]):
        raise PermissionError("access denied")

    monkeypatch.setattr(os, "scandir", denied_scandir)
    result = ToolDispatcher(create_read_only_registry(Workspace(tmp_path))).dispatch(
        ToolCall(id="call_1", name="list_files", arguments={})
    )

    assert isinstance(result, ToolResult)
    assert result.tool_call_id == "call_1"
    assert result.is_error is True
    assert "Permission denied" in result.content


def test_read_file_reads_utf8_text(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("你好，workspace", encoding="utf-8")

    assert Workspace(tmp_path).read_file("notes.txt") == "你好，workspace"


def test_read_file_marks_output_when_character_limit_is_reached(tmp_path: Path) -> None:
    (tmp_path / "long.txt").write_text("0123456789", encoding="utf-8")

    assert Workspace(tmp_path).read_file("long.txt", max_output_chars=5) == (
        "01234\n...[output truncated]"
    )


def test_read_file_reads_only_a_bounded_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = "x" * 100_000
    (tmp_path / "large.txt").write_text(content, encoding="utf-8")
    original_open = Path.open
    bytes_read = 0

    class TrackingFile:
        def __init__(self, file_handle):
            self.file_handle = file_handle

        def __enter__(self):
            self.file_handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.file_handle.__exit__(*args)

        def read(self, size: int = -1):
            nonlocal bytes_read
            chunk = self.file_handle.read(size)
            bytes_read += len(chunk)
            return chunk

    def tracking_open(path: Path, *args, **kwargs):
        return TrackingFile(original_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", tracking_open)
    result = Workspace(tmp_path).read_file("large.txt", max_output_chars=10)

    assert result == "xxxxxxxxxx\n...[output truncated]"
    assert bytes_read < len(content)


def test_read_file_preserves_utf8_character_boundaries(tmp_path: Path) -> None:
    (tmp_path / "unicode.txt").write_text("你好世界", encoding="utf-8")

    result = Workspace(tmp_path).read_file("unicode.txt", max_output_chars=3)

    assert result == "你好世\n...[output truncated]"


@pytest.mark.parametrize(
    ("filename", "expected_error"),
    [
        ("missing.txt", "does not exist"),
        ("directory", "not a file"),
    ],
)
def test_read_file_rejects_missing_or_directory_paths(
    tmp_path: Path,
    filename: str,
    expected_error: str,
) -> None:
    (tmp_path / "directory").mkdir()

    result = ToolDispatcher(create_read_only_registry(Workspace(tmp_path))).dispatch(
        ToolCall(id="call_1", name="read_file", arguments={"path": filename})
    )

    assert result.is_error is True
    assert expected_error in result.content


@pytest.mark.parametrize("content", [b"\x00\x01\x02", b"\xff\xfe\xfd"])
def test_read_file_rejects_binary_or_invalid_utf8(
    tmp_path: Path,
    content: bytes,
) -> None:
    (tmp_path / "data.bin").write_bytes(content)

    with pytest.raises(WorkspaceFileError, match="binary|UTF-8"):
        Workspace(tmp_path).read_file("data.bin")


def test_read_only_registry_dispatches_workspace_tools(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Project README", encoding="utf-8")
    dispatcher = ToolDispatcher(create_read_only_registry(Workspace(tmp_path)))

    listed = dispatcher.dispatch(ToolCall(id="call_1", name="list_files", arguments={}))
    read = dispatcher.dispatch(
        ToolCall(id="call_2", name="read_file", arguments={"path": "README.md"})
    )

    assert listed.content == "README.md"
    assert read.content == "Project README"
    assert listed.is_error is False
    assert read.is_error is False


def test_read_only_tool_argument_models_expose_expected_defaults() -> None:
    assert ListFilesArguments().path == "."
    assert ReadFileArguments(path="README.md").path == "README.md"


@pytest.mark.parametrize("path", ["..", "../outside.txt", "../../outside.txt"])
def test_workspace_rejects_parent_escapes(tmp_path: Path, path: str) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError, match="escapes|relative"):
        workspace.resolve_path(path)


@pytest.mark.parametrize("path", [Path("C:/outside.txt"), "C:/outside.txt", ""])
def test_workspace_rejects_absolute_paths(tmp_path: Path, path: str | Path) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError, match="relative|empty"):
        workspace.resolve_path(path)


def test_workspace_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(WorkspacePathError, match="escapes"):
        Workspace(tmp_path).resolve_path("link/file.txt")
