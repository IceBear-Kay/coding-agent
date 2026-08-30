import json
import os
from pathlib import Path

import pytest

from coding_agent.approval import ApprovalRequest
from coding_agent.file_tools import (
    DEFAULT_MAX_FILE_BYTES,
    EditFileArguments,
    WriteFileArguments,
    create_workspace_registry,
    edit_file_tool_spec,
    write_file_tool_spec,
)
from coding_agent.models import ToolCall
from coding_agent.tools import ToolDispatcher, ToolRegistry, Workspace


def dispatch_write(
    workspace: Workspace,
    path: str,
    content: str,
    approval_callback=None,
    *,
    max_content_bytes: int = DEFAULT_MAX_FILE_BYTES,
):
    spec = write_file_tool_spec(
        workspace,
        approval_callback,
        max_content_bytes=max_content_bytes,
    )
    return ToolDispatcher(ToolRegistry([spec])).dispatch(
        ToolCall(
            id="call_write",
            name="write_file",
            arguments={"path": path, "content": content},
        )
    )


def dispatch_edit(
    workspace: Workspace,
    path: str,
    old_text: str,
    new_text: str,
    approval_callback=None,
    *,
    max_content_bytes: int = DEFAULT_MAX_FILE_BYTES,
):
    spec = edit_file_tool_spec(
        workspace,
        approval_callback,
        max_content_bytes=max_content_bytes,
    )
    return ToolDispatcher(ToolRegistry([spec])).dispatch(
        ToolCall(
            id="call_edit",
            name="edit_file",
            arguments={"path": path, "old_text": old_text, "new_text": new_text},
        )
    )


def test_write_file_creates_approved_utf8_file_and_missing_parents(tmp_path: Path) -> None:
    content = "第一行\r\nsecond line\n"

    result = dispatch_write(Workspace(tmp_path), "src/generated.txt", content, lambda _: True)

    assert result.tool_call_id == "call_write"
    assert result.is_error is False
    assert json.loads(result.content) == {
        "status": "created",
        "path": "src/generated.txt",
        "bytes_written": len(content.encode("utf-8")),
    }
    assert (tmp_path / "src" / "generated.txt").read_bytes() == content.encode("utf-8")


def test_write_file_sends_exact_immutable_preview_for_approval(tmp_path: Path) -> None:
    requests: list[ApprovalRequest] = []
    content = "hello\n世界\x1b"

    def reject(request: ApprovalRequest) -> bool:
        requests.append(request)
        return False

    result = dispatch_write(Workspace(tmp_path), "new/note.txt", content, reject)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "denied"
    assert len(requests) == 1
    assert requests[0].operation == "write_file"
    assert "Path: new/note.txt" in requests[0].preview
    assert "Parent directories to create: new" in requests[0].preview
    assert f"Content: {json.dumps(content, ensure_ascii=False)}" in requests[0].preview
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("approval_callback", [None, lambda _: False])
def test_write_file_denial_does_not_create_file_or_parent_directories(
    tmp_path: Path,
    approval_callback,
) -> None:
    result = dispatch_write(
        Workspace(tmp_path),
        "one/two/denied.txt",
        "content",
        approval_callback,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "denied"
    assert list(tmp_path.iterdir()) == []


def test_write_file_never_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("original", encoding="utf-8")
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    result = dispatch_write(Workspace(tmp_path), "existing.txt", "replacement", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "already_exists"
    assert target.read_text(encoding="utf-8") == "original"
    assert approval_calls == 0


def test_write_file_rejects_existing_directory(tmp_path: Path) -> None:
    (tmp_path / "target").mkdir()

    result = dispatch_write(Workspace(tmp_path), "target", "content", lambda _: True)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "already_exists"


def test_write_file_detects_target_created_during_approval(tmp_path: Path) -> None:
    target = tmp_path / "raced.txt"

    def approve(_: ApprovalRequest) -> bool:
        target.write_text("created concurrently", encoding="utf-8")
        return True

    result = dispatch_write(Workspace(tmp_path), "raced.txt", "agent content", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "conflict"
    assert target.read_text(encoding="utf-8") == "created concurrently"


def test_write_file_detects_parent_created_during_approval(tmp_path: Path) -> None:
    parent = tmp_path / "new-parent"

    def approve(_: ApprovalRequest) -> bool:
        parent.mkdir()
        return True

    result = dispatch_write(
        Workspace(tmp_path),
        "new-parent/file.txt",
        "content",
        approve,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "conflict"
    assert parent.is_dir()
    assert not (parent / "file.txt").exists()


def test_write_file_detects_existing_parent_replaced_during_approval(tmp_path: Path) -> None:
    parent = tmp_path / "existing-parent"
    parent.mkdir()

    def approve(_: ApprovalRequest) -> bool:
        parent.rmdir()
        parent.mkdir()
        return True

    result = dispatch_write(
        Workspace(tmp_path),
        "existing-parent/file.txt",
        "content",
        approve,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "conflict"
    assert not (parent / "file.txt").exists()


def test_write_file_detects_workspace_root_replaced_during_approval(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    moved_root = tmp_path / "moved-workspace"
    workspace = Workspace(workspace_root)

    def approve(_: ApprovalRequest) -> bool:
        workspace_root.rename(moved_root)
        workspace_root.mkdir()
        return True

    result = dispatch_write(workspace, "created.txt", "content", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "conflict"
    assert not (workspace_root / "created.txt").exists()
    assert not (moved_root / "created.txt").exists()


def test_write_file_detects_workspace_root_replaced_with_external_symlink(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    moved_root = tmp_path / "moved-workspace"
    external = tmp_path / "external"
    external.mkdir()
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    probe.unlink()
    workspace = Workspace(workspace_root)

    def approve(_: ApprovalRequest) -> bool:
        workspace_root.rename(moved_root)
        workspace_root.symlink_to(external, target_is_directory=True)
        return True

    result = dispatch_write(workspace, "escaped.txt", "content", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "conflict"
    assert not (external / "escaped.txt").exists()
    assert not (moved_root / "escaped.txt").exists()


def test_write_file_enforces_utf8_byte_limit_before_approval(tmp_path: Path) -> None:
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    result = dispatch_write(
        Workspace(tmp_path),
        "too-large.txt",
        "你好",
        approve,
        max_content_bytes=5,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "too_large"
    assert approval_calls == 0
    assert not (tmp_path / "too-large.txt").exists()


def test_write_file_accepts_content_at_exact_utf8_byte_limit(tmp_path: Path) -> None:
    result = dispatch_write(
        Workspace(tmp_path),
        "boundary.txt",
        "你a",
        lambda _: True,
        max_content_bytes=4,
    )

    assert result.is_error is False
    assert (tmp_path / "boundary.txt").read_bytes() == "你a".encode()


def test_write_file_can_create_empty_file(tmp_path: Path) -> None:
    result = dispatch_write(Workspace(tmp_path), "empty.txt", "", lambda _: True)

    assert result.is_error is False
    assert (tmp_path / "empty.txt").read_bytes() == b""


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "C:/outside.txt",
        "//server/share/file.txt",
        ".git/config",
        ".local/private.txt",
        ".venv/config.txt",
        ".env",
        ".env.production",
        ".env/secret.txt",
        ".env.example/secret.txt",
        "config/.env.local",
        "file.txt:stream",
        "NUL.txt",
        "bad?.txt",
        "trailing-dot.",
        "control\x00.txt",
    ],
)
def test_write_file_rejects_unsafe_paths_without_approval(tmp_path: Path, path: str) -> None:
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    result = dispatch_write(Workspace(tmp_path), path, "content", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "invalid_path"
    assert approval_calls == 0


def test_write_file_allows_environment_template(tmp_path: Path) -> None:
    result = dispatch_write(Workspace(tmp_path), ".env.example", "TOKEN=", lambda _: True)

    assert result.is_error is False
    assert (tmp_path / ".env.example").read_text(encoding="utf-8") == "TOKEN="


def test_write_file_approval_failure_makes_no_filesystem_changes(tmp_path: Path) -> None:
    def broken_approval(_: ApprovalRequest) -> bool:
        raise EOFError("input closed")

    result = dispatch_write(
        Workspace(tmp_path),
        "new/file.txt",
        "content",
        broken_approval,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "approval_failed"
    assert list(tmp_path.iterdir()) == []


def test_write_file_reports_path_inspection_failure_without_requesting_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace(tmp_path)
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    def denied_lstat(_: Path):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "lstat", denied_lstat)
    result = dispatch_write(workspace, "blocked.txt", "content", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "failed"
    assert approval_calls == 0


def test_write_file_rejects_reparse_parent_before_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "linked"
    parent.mkdir()
    original_lstat = Path.lstat

    def reparse_lstat(path: Path):
        result = original_lstat(path)
        if path == parent:

            class ReparseStat:
                st_mode = result.st_mode
                st_reparse_tag = 1

            return ReparseStat()
        return result

    monkeypatch.setattr(Path, "lstat", reparse_lstat)
    result = dispatch_write(Workspace(tmp_path), "linked/file.txt", "content", lambda _: True)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "invalid_path"
    assert not (parent / "file.txt").exists()


def test_write_file_rejects_symbolic_link_parent(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(actual, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    result = dispatch_write(Workspace(tmp_path), "linked/file.txt", "content", lambda _: True)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "invalid_path"
    assert not (actual / "file.txt").exists()


def test_write_file_cleans_new_directories_after_open_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open

    def failing_open(path: Path, *args, **kwargs):
        if path.name == "failed.txt":
            raise PermissionError("denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    result = dispatch_write(
        Workspace(tmp_path),
        "new/failed.txt",
        "content",
        lambda _: True,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "failed"
    assert not (tmp_path / "new").exists()


def test_write_file_cleans_partial_file_and_new_directories_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open

    class InterruptingFile:
        def __init__(self, file_handle):
            self.file_handle = file_handle

        def __enter__(self):
            self.file_handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.file_handle.__exit__(*args)

        def write(self, content: bytes) -> int:
            self.file_handle.write(content[:3])
            self.file_handle.flush()
            raise KeyboardInterrupt

    def interrupting_open(path: Path, *args, **kwargs):
        file_handle = original_open(path, *args, **kwargs)
        if path.name == "partial.txt" and args == ("xb",):
            return InterruptingFile(file_handle)
        return file_handle

    monkeypatch.setattr(Path, "open", interrupting_open)

    with pytest.raises(KeyboardInterrupt):
        dispatch_write(
            Workspace(tmp_path),
            "new/partial.txt",
            "complete content",
            lambda _: True,
        )

    assert not (tmp_path / "new").exists()


def test_write_file_arguments_forbid_extra_fields() -> None:
    spec = write_file_tool_spec(Workspace(Path.cwd()), lambda _: True)
    result = ToolDispatcher(ToolRegistry([spec])).dispatch(
        ToolCall(
            id="call_write",
            name="write_file",
            arguments={"path": "file.txt", "content": "content", "overwrite": True},
        )
    )

    assert result.is_error is True
    assert "unexpected" in result.content
    assert WriteFileArguments(path="file.txt", content="").content == ""


def test_edit_file_replaces_one_exact_fragment_and_preserves_line_endings(
    tmp_path: Path,
) -> None:
    target = tmp_path / "program.py"
    original = "title = '示例'\r\ncount = 1\r\n"
    expected = "title = '示例'\r\ncount = 2\r\n"
    target.write_bytes(original.encode("utf-8"))

    result = dispatch_edit(
        Workspace(tmp_path),
        "program.py",
        "count = 1",
        "count = 2",
        lambda _: True,
    )

    assert result.tool_call_id == "call_edit"
    assert result.is_error is False
    assert json.loads(result.content) == {
        "status": "edited",
        "path": "program.py",
        "replacements": 1,
        "bytes_written": len(expected.encode("utf-8")),
    }
    assert target.read_bytes() == expected.encode("utf-8")


def test_edit_file_succeeds_in_existing_nested_directory(tmp_path: Path) -> None:
    parent = tmp_path / "nested"
    parent.mkdir()
    target = parent / "notes.txt"
    target.write_text("before", encoding="utf-8")

    result = dispatch_edit(
        Workspace(tmp_path),
        "nested/notes.txt",
        "before",
        "after",
        lambda _: True,
    )

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "after"


def test_edit_file_sends_exact_diff_preview_before_changing_file(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    original = "alpha\r\nbeta\r\n"
    target.write_bytes(original.encode())
    requests: list[ApprovalRequest] = []

    def reject(request: ApprovalRequest) -> bool:
        requests.append(request)
        return False

    result = dispatch_edit(Workspace(tmp_path), "notes.txt", "beta", "gamma", reject)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "denied"
    assert target.read_bytes() == original.encode()
    assert len(requests) == 1
    assert requests[0].operation == "edit_file"
    assert "Path: notes.txt" in requests[0].preview
    assert json.dumps("-beta\r\n") in requests[0].preview
    assert json.dumps("+gamma\r\n") in requests[0].preview


def test_edit_file_rejects_overlapping_multiple_matches_without_approval(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("aaa", encoding="utf-8")
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    result = dispatch_edit(Workspace(tmp_path), "notes.txt", "aa", "X", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "multiple_matches"
    assert approval_calls == 0
    assert target.read_text(encoding="utf-8") == "aaa"


def test_edit_file_allows_one_match_when_no_overlap_exists(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("aab", encoding="utf-8")

    result = dispatch_edit(Workspace(tmp_path), "notes.txt", "aa", "X", lambda _: True)

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "Xb"


@pytest.mark.parametrize("approval_callback", [None, lambda _: False])
def test_edit_file_denial_preserves_original_file(
    tmp_path: Path,
    approval_callback,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")

    result = dispatch_edit(
        Workspace(tmp_path),
        "notes.txt",
        "before",
        "after",
        approval_callback,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "denied"
    assert target.read_text(encoding="utf-8") == "before"


@pytest.mark.parametrize(
    ("old_text", "new_text", "expected_status"),
    [
        ("missing", "new", "no_match"),
        ("same", "new", "multiple_matches"),
        ("and", "and", "no_change"),
    ],
)
def test_edit_file_rejects_ambiguous_or_noop_replacements_before_approval(
    tmp_path: Path,
    old_text: str,
    new_text: str,
    expected_status: str,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("same and same", encoding="utf-8")
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    result = dispatch_edit(Workspace(tmp_path), "notes.txt", old_text, new_text, approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == expected_status
    assert approval_calls == 0
    assert target.read_text(encoding="utf-8") == "same and same"


def test_edit_file_allows_deleting_the_selected_fragment(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("keep REMOVE keep", encoding="utf-8")

    result = dispatch_edit(
        Workspace(tmp_path),
        "notes.txt",
        " REMOVE",
        "",
        lambda _: True,
    )

    assert result.is_error is False
    assert target.read_text(encoding="utf-8") == "keep keep"


def test_edit_file_rejects_empty_old_text_as_invalid_arguments(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("unchanged", encoding="utf-8")

    result = dispatch_edit(Workspace(tmp_path), "notes.txt", "", "new", lambda _: True)

    assert result.is_error is True
    assert "Invalid arguments" in result.content
    assert target.read_text(encoding="utf-8") == "unchanged"


def test_edit_file_reads_only_limit_plus_one_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "large.txt"
    target.write_bytes(b"x" * 100)
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
    result = dispatch_edit(
        Workspace(tmp_path),
        "large.txt",
        "x",
        "y",
        lambda _: True,
        max_content_bytes=5,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "too_large"
    assert bytes_read == 6


def test_edit_file_rejects_result_over_utf8_byte_limit(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("a", encoding="utf-8")

    result = dispatch_edit(
        Workspace(tmp_path),
        "notes.txt",
        "a",
        "你好",
        lambda _: True,
        max_content_bytes=5,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "too_large"
    assert target.read_text(encoding="utf-8") == "a"


@pytest.mark.parametrize("content", [b"binary\x00data", b"\xff\xfe"])
def test_edit_file_rejects_binary_or_invalid_utf8(tmp_path: Path, content: bytes) -> None:
    target = tmp_path / "data.txt"
    target.write_bytes(content)

    result = dispatch_edit(Workspace(tmp_path), "data.txt", "data", "new", lambda _: True)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "invalid_content"
    assert target.read_bytes() == content


@pytest.mark.parametrize(
    ("path", "expected_status"),
    [("missing.txt", "not_found"), ("directory", "invalid_path")],
)
def test_edit_file_rejects_missing_or_non_file_target(
    tmp_path: Path,
    path: str,
    expected_status: str,
) -> None:
    (tmp_path / "directory").mkdir()

    result = dispatch_edit(Workspace(tmp_path), path, "old", "new", lambda _: True)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == expected_status


def test_edit_file_rejects_hard_link_target(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    alias = tmp_path / "alias.txt"
    target.write_text("original", encoding="utf-8")
    try:
        os.link(target, alias)
    except OSError:
        pytest.skip("hard link creation is unavailable")

    result = dispatch_edit(Workspace(tmp_path), "target.txt", "original", "changed", lambda _: True)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "invalid_path"
    assert target.read_text(encoding="utf-8") == "original"
    assert alias.read_text(encoding="utf-8") == "original"


def test_edit_file_rejects_reparse_target_before_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("original", encoding="utf-8")
    original_lstat = Path.lstat
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    def reparse_lstat(path: Path):
        result = original_lstat(path)
        if path == target:

            class ReparseStat:
                st_mode = result.st_mode
                st_reparse_tag = 1

            return ReparseStat()
        return result

    monkeypatch.setattr(Path, "lstat", reparse_lstat)
    result = dispatch_edit(Workspace(tmp_path), "target.txt", "original", "changed", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "invalid_path"
    assert approval_calls == 0


def test_edit_file_reports_read_permission_failure_before_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("original", encoding="utf-8")
    workspace = Workspace(tmp_path)
    original_open = Path.open
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    def denied_open(path: Path, *args, **kwargs):
        if path == target:
            raise PermissionError("denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    result = dispatch_edit(workspace, "target.txt", "original", "changed", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "failed"
    assert approval_calls == 0
    with original_open(target, "r", encoding="utf-8") as file_handle:
        assert file_handle.read() == "original"


def test_edit_file_detects_content_changed_during_approval(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")

    def approve(_: ApprovalRequest) -> bool:
        target.write_text("changed by user", encoding="utf-8")
        return True

    result = dispatch_edit(Workspace(tmp_path), "notes.txt", "before", "after", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "conflict"
    assert target.read_text(encoding="utf-8") == "changed by user"


def test_edit_file_detects_target_deleted_during_approval(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")

    def approve(_: ApprovalRequest) -> bool:
        target.unlink()
        return True

    result = dispatch_edit(Workspace(tmp_path), "notes.txt", "before", "after", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "conflict"
    assert not target.exists()


def test_edit_file_detects_target_replaced_during_approval(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")

    def approve(_: ApprovalRequest) -> bool:
        target.unlink()
        target.write_text("before", encoding="utf-8")
        return True

    result = dispatch_edit(Workspace(tmp_path), "notes.txt", "before", "after", approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "conflict"
    assert target.read_text(encoding="utf-8") == "before"


def test_edit_file_replace_failure_preserves_original_and_cleans_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")

    def failing_replace(_: Path, __: Path) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(os, "replace", failing_replace)
    result = dispatch_edit(
        Workspace(tmp_path),
        "notes.txt",
        "before",
        "after",
        lambda _: True,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "failed"
    assert target.read_text(encoding="utf-8") == "before"
    assert list(tmp_path.glob(".notes.txt.*.tmp")) == []


def test_edit_file_cleans_temporary_file_and_preserves_original_on_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")

    def interrupt_fsync(_: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "fsync", interrupt_fsync)

    with pytest.raises(KeyboardInterrupt):
        dispatch_edit(Workspace(tmp_path), "notes.txt", "before", "after", lambda _: True)

    assert target.read_text(encoding="utf-8") == "before"
    assert list(tmp_path.glob(".notes.txt.*.tmp")) == []


def test_edit_file_arguments_forbid_extra_fields(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    spec = edit_file_tool_spec(Workspace(tmp_path), lambda _: True)
    result = ToolDispatcher(ToolRegistry([spec])).dispatch(
        ToolCall(
            id="call_edit",
            name="edit_file",
            arguments={
                "path": "notes.txt",
                "old_text": "before",
                "new_text": "after",
                "all": True,
            },
        )
    )

    assert result.is_error is True
    assert "unexpected" in result.content
    assert EditFileArguments(path="notes.txt", old_text="a", new_text="").new_text == ""


def test_workspace_registry_only_exposes_mutations_when_enabled(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    read_only_names = [spec.name for spec in create_workspace_registry(workspace)]
    writable_names = [spec.name for spec in create_workspace_registry(workspace, allow_write=True)]

    assert read_only_names == ["list_files", "read_file"]
    assert writable_names == ["list_files", "read_file", "write_file", "edit_file"]
