"""Approved workspace file mutation tools."""

import difflib
import hashlib
import json
import os
import stat
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from os import stat_result
from pathlib import Path, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field

from coding_agent.approval import ApprovalCallback, ApprovalRequest, request_approval
from coding_agent.command_tools import CommandLimits, run_command_tool_spec
from coding_agent.tools import (
    ToolOutput,
    ToolRegistry,
    ToolSpec,
    Workspace,
    create_read_only_registry,
)

DEFAULT_MAX_FILE_BYTES = 65_536
_PROTECTED_DIRECTORY_NAMES = frozenset({".git", ".local", ".venv"})
_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_WINDOWS_REPARSE_POINT = 0x0400


class WriteFileArguments(BaseModel):
    """Arguments accepted by the write_file tool."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    content: str


class EditFileArguments(BaseModel):
    """Arguments accepted by the edit_file tool."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str


class _UnsafeWritePathError(ValueError):
    """The requested write target is unsafe or unsupported."""


class _WritePathInspectionError(OSError):
    """The requested write target could not be inspected."""


class _TargetAlreadyExistsError(FileExistsError):
    """The requested write target already exists."""

    def __init__(self, relative_path: str) -> None:
        super().__init__(relative_path)
        self.relative_path = relative_path


class _TargetNotFoundError(FileNotFoundError):
    """The requested edit target does not exist."""


class _FileTooLargeError(ValueError):
    """The source or resulting file exceeds the configured byte limit."""


class _InvalidTextFileError(ValueError):
    """The edit target is not supported UTF-8 text."""


class _FileReadError(OSError):
    """The edit target could not be read."""


class _EditConflictError(RuntimeError):
    """The edit target changed while an operation was being prepared."""


class _TextMatchError(ValueError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class _ParentFingerprint:
    path: Path
    device: int
    inode: int
    mode: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _WritePlan:
    relative_path: str
    target: Path
    content: bytes
    missing_parents: tuple[Path, ...]
    existing_parents: tuple[_ParentFingerprint, ...]
    preview: str


@dataclass(frozen=True, slots=True)
class _TargetFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    mode: int
    digest: str


@dataclass(frozen=True, slots=True)
class _EditSource:
    relative_path: str
    parts: tuple[str, ...]
    target: Path
    text: str
    parents: tuple[_ParentFingerprint, ...]
    fingerprint: _TargetFingerprint


@dataclass(frozen=True, slots=True)
class _EditPlan:
    relative_path: str
    parts: tuple[str, ...]
    target: Path
    new_content: bytes
    parents: tuple[_ParentFingerprint, ...]
    source_fingerprint: _TargetFingerprint
    preview: str


class _WriteFileHandler:
    def __init__(
        self,
        workspace: Workspace,
        approval_callback: ApprovalCallback | None,
        max_content_bytes: int,
    ) -> None:
        if max_content_bytes < 0:
            raise ValueError("max_content_bytes must not be negative")
        self._workspace = workspace
        self._approval_callback = approval_callback
        self._max_content_bytes = max_content_bytes

    def __call__(self, arguments: WriteFileArguments) -> ToolOutput:
        try:
            plan = self._prepare(arguments)
        except _TargetAlreadyExistsError as exc:
            return self._error(
                "already_exists",
                exc.relative_path,
                "Target already exists; write_file never overwrites existing objects.",
            )
        except _UnsafeWritePathError as exc:
            return self._error("invalid_path", arguments.path, str(exc))
        except _WritePathInspectionError as exc:
            return self._error("failed", arguments.path, str(exc))

        if len(plan.content) > self._max_content_bytes:
            return self._error(
                "too_large",
                plan.relative_path,
                f"UTF-8 content exceeds the {self._max_content_bytes}-byte limit.",
            )

        request = ApprovalRequest(operation="write_file", preview=plan.preview)
        try:
            approved = request_approval(request, self._approval_callback)
        except Exception as exc:
            return self._error(
                "approval_failed",
                plan.relative_path,
                f"Approval could not be obtained: {exc}",
            )
        if not approved:
            return self._error(
                "denied",
                plan.relative_path,
                "Operation was not approved and no filesystem changes were made.",
            )

        try:
            verified_plan = self._prepare(arguments)
        except (
            _TargetAlreadyExistsError,
            _UnsafeWritePathError,
            _WritePathInspectionError,
        ) as exc:
            return self._error(
                "conflict",
                plan.relative_path,
                f"Workspace state changed after approval: {exc}",
            )
        if (
            verified_plan.missing_parents != plan.missing_parents
            or verified_plan.existing_parents != plan.existing_parents
        ):
            return self._error(
                "conflict",
                plan.relative_path,
                "Workspace parent directories changed after approval.",
            )

        return self._execute(plan)

    def _prepare(self, arguments: WriteFileArguments) -> _WritePlan:
        relative_path, parts = _normalize_write_path(arguments.path)
        target = self._workspace.root.joinpath(*parts)
        missing_parents, existing_parents = _inspect_parent_path(
            self._workspace.root,
            parts,
        )

        target_stat = _lstat(target)
        if target_stat is not None:
            raise _TargetAlreadyExistsError(relative_path)

        content = arguments.content.encode("utf-8")
        preview = _build_write_preview(
            relative_path,
            arguments.content,
            content_bytes=len(content),
            missing_parents=tuple(
                directory.relative_to(self._workspace.root).as_posix()
                for directory in missing_parents
            ),
        )
        return _WritePlan(
            relative_path=relative_path,
            target=target,
            content=content,
            missing_parents=tuple(missing_parents),
            existing_parents=tuple(existing_parents),
            preview=preview,
        )

    def _execute(self, plan: _WritePlan) -> ToolOutput:
        created_directories: list[Path] = []
        target_created = False
        try:
            for fingerprint in plan.existing_parents:
                parent_stat = _lstat(fingerprint.path)
                if not _matches_parent_fingerprint(fingerprint, parent_stat):
                    raise _UnsafeWritePathError("Write path parent changed before file creation.")

            for directory in plan.missing_parents:
                directory.mkdir()
                created_directories.append(directory)

            for parent in plan.target.parents:
                if parent == self._workspace.root:
                    break
                parent_stat = _lstat(parent)
                if (
                    parent_stat is None
                    or _is_reparse_point(parent_stat)
                    or not stat.S_ISDIR(parent_stat.st_mode)
                ):
                    raise _UnsafeWritePathError("Write path parent changed before file creation.")

            with plan.target.open("xb") as file_handle:
                target_created = True
                bytes_written = file_handle.write(plan.content)
                if bytes_written != len(plan.content):
                    raise OSError("Unable to write the complete file content")
        except KeyboardInterrupt:
            _cleanup_created_paths(
                plan.target if target_created else None,
                created_directories,
            )
            raise
        except FileExistsError:
            cleanup_incomplete = _cleanup_created_paths(
                plan.target if target_created else None,
                created_directories,
            )
            return self._error(
                "conflict",
                plan.relative_path,
                "Target or parent path changed before file creation.",
                cleanup_incomplete=cleanup_incomplete,
            )
        except (OSError, _UnsafeWritePathError) as exc:
            cleanup_incomplete = _cleanup_created_paths(
                plan.target if target_created else None,
                created_directories,
            )
            return self._error(
                "failed",
                plan.relative_path,
                f"File creation failed: {exc}",
                cleanup_incomplete=cleanup_incomplete,
            )

        return ToolOutput(
            status="created",
            details={
                "path": plan.relative_path,
                "bytes_written": len(plan.content),
            },
        )

    @staticmethod
    def _error(
        status: str,
        path: str,
        message: str,
        *,
        cleanup_incomplete: tuple[str, ...] = (),
    ) -> ToolOutput:
        details: dict[str, str | list[str]] = {"path": path, "message": message}
        if cleanup_incomplete:
            details["cleanup_incomplete"] = list(cleanup_incomplete)
        return ToolOutput(status=status, details=details, is_error=True)


class _EditFileHandler:
    def __init__(
        self,
        workspace: Workspace,
        approval_callback: ApprovalCallback | None,
        max_content_bytes: int,
    ) -> None:
        if max_content_bytes < 0:
            raise ValueError("max_content_bytes must not be negative")
        self._workspace = workspace
        self._approval_callback = approval_callback
        self._max_content_bytes = max_content_bytes

    def __call__(self, arguments: EditFileArguments) -> ToolOutput:
        try:
            plan = self._prepare(arguments)
        except _TargetNotFoundError:
            return self._error("not_found", arguments.path, "Edit target does not exist.")
        except _UnsafeWritePathError as exc:
            return self._error("invalid_path", arguments.path, str(exc))
        except _FileTooLargeError as exc:
            return self._error("too_large", arguments.path, str(exc))
        except _InvalidTextFileError as exc:
            return self._error("invalid_content", arguments.path, str(exc))
        except _TextMatchError as exc:
            return self._error(exc.status, arguments.path, str(exc))
        except (_WritePathInspectionError, _FileReadError) as exc:
            return self._error("failed", arguments.path, str(exc))
        except _EditConflictError as exc:
            return self._error("conflict", arguments.path, str(exc))

        request = ApprovalRequest(operation="edit_file", preview=plan.preview)
        try:
            approved = request_approval(request, self._approval_callback)
        except Exception as exc:
            return self._error(
                "approval_failed",
                plan.relative_path,
                f"Approval could not be obtained: {exc}",
            )
        if not approved:
            return self._error(
                "denied",
                plan.relative_path,
                "Operation was not approved and the file was not changed.",
            )

        try:
            verified_plan = self._prepare(arguments)
        except (
            _TargetNotFoundError,
            _UnsafeWritePathError,
            _FileTooLargeError,
            _InvalidTextFileError,
            _TextMatchError,
            _WritePathInspectionError,
            _FileReadError,
            _EditConflictError,
        ) as exc:
            return self._error(
                "conflict",
                plan.relative_path,
                f"Workspace state changed after approval: {exc}",
            )
        if (
            verified_plan.parents != plan.parents
            or verified_plan.source_fingerprint != plan.source_fingerprint
            or verified_plan.new_content != plan.new_content
        ):
            return self._error(
                "conflict",
                plan.relative_path,
                "Edit target changed after approval.",
            )

        return self._execute(plan)

    def _prepare(self, arguments: EditFileArguments) -> _EditPlan:
        source = _load_edit_source(
            self._workspace,
            arguments.path,
            self._max_content_bytes,
        )
        match_count = _count_overlapping_matches(source.text, arguments.old_text)
        if match_count == 0:
            raise _TextMatchError("no_match", "old_text was not found in the target file.")
        if match_count > 1:
            raise _TextMatchError(
                "multiple_matches",
                "old_text must match exactly once in the target file.",
            )

        new_text = source.text.replace(arguments.old_text, arguments.new_text, 1)
        if new_text == source.text:
            raise _TextMatchError(
                "no_change", "The requested replacement would not change the file."
            )
        new_content = new_text.encode("utf-8")
        if len(new_content) > self._max_content_bytes:
            raise _FileTooLargeError(
                f"Edited UTF-8 content exceeds the {self._max_content_bytes}-byte limit."
            )

        return _EditPlan(
            relative_path=source.relative_path,
            parts=source.parts,
            target=source.target,
            new_content=new_content,
            parents=source.parents,
            source_fingerprint=source.fingerprint,
            preview=_build_edit_preview(source.relative_path, source.text, new_text),
        )

    def _execute(self, plan: _EditPlan) -> ToolOutput:
        descriptor: int | None = None
        temporary_path: Path | None = None
        try:
            self._assert_current(plan)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{plan.target.name}.",
                suffix=".tmp",
                dir=plan.target.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as file_handle:
                descriptor = None
                bytes_written = file_handle.write(plan.new_content)
                if bytes_written != len(plan.new_content):
                    raise OSError("Unable to write the complete edited content")
                file_handle.flush()
                os.fsync(file_handle.fileno())

            os.chmod(temporary_path, stat.S_IMODE(plan.source_fingerprint.mode))
            self._assert_current(plan, parent_identity_only=True)
            os.replace(temporary_path, plan.target)
            temporary_path = None
        except KeyboardInterrupt:
            _close_file_descriptor(descriptor)
            descriptor = None
            _cleanup_temporary_file(temporary_path, self._workspace.root)
            raise
        except _EditConflictError as exc:
            _close_file_descriptor(descriptor)
            descriptor = None
            cleanup_incomplete = _cleanup_temporary_file(temporary_path, self._workspace.root)
            return self._error(
                "conflict",
                plan.relative_path,
                str(exc),
                cleanup_incomplete=cleanup_incomplete,
            )
        except OSError as exc:
            _close_file_descriptor(descriptor)
            descriptor = None
            cleanup_incomplete = _cleanup_temporary_file(temporary_path, self._workspace.root)
            return self._error(
                "failed",
                plan.relative_path,
                f"File edit failed: {exc}",
                cleanup_incomplete=cleanup_incomplete,
            )
        finally:
            _close_file_descriptor(descriptor)

        return ToolOutput(
            status="edited",
            details={
                "path": plan.relative_path,
                "replacements": 1,
                "bytes_written": len(plan.new_content),
            },
        )

    def _assert_current(
        self,
        plan: _EditPlan,
        *,
        parent_identity_only: bool = False,
    ) -> None:
        try:
            source = _load_edit_source(
                self._workspace,
                plan.relative_path,
                self._max_content_bytes,
            )
        except (
            _TargetNotFoundError,
            _UnsafeWritePathError,
            _FileTooLargeError,
            _InvalidTextFileError,
            _WritePathInspectionError,
            _FileReadError,
            _EditConflictError,
        ) as exc:
            raise _EditConflictError(f"Edit target changed before replacement: {exc}") from exc
        parents_match = (
            _same_parent_identities(source.parents, plan.parents)
            if parent_identity_only
            else source.parents == plan.parents
        )
        if not parents_match or source.fingerprint != plan.source_fingerprint:
            raise _EditConflictError("Edit target changed before replacement.")

    @staticmethod
    def _error(
        status: str,
        path: str,
        message: str,
        *,
        cleanup_incomplete: tuple[str, ...] = (),
    ) -> ToolOutput:
        details: dict[str, str | list[str]] = {"path": path, "message": message}
        if cleanup_incomplete:
            details["cleanup_incomplete"] = list(cleanup_incomplete)
        return ToolOutput(status=status, details=details, is_error=True)


def write_file_tool_spec(
    workspace: Workspace,
    approval_callback: ApprovalCallback | None = None,
    *,
    max_content_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> ToolSpec[WriteFileArguments]:
    """Build the approved write_file tool specification."""
    return ToolSpec(
        name="write_file",
        description="Create a new UTF-8 text file inside the workspace after approval.",
        parameters=WriteFileArguments,
        handler=_WriteFileHandler(workspace, approval_callback, max_content_bytes),
    )


def edit_file_tool_spec(
    workspace: Workspace,
    approval_callback: ApprovalCallback | None = None,
    *,
    max_content_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> ToolSpec[EditFileArguments]:
    """Build the approved edit_file tool specification."""
    return ToolSpec(
        name="edit_file",
        description="Replace one exact text fragment in a workspace file after approval.",
        parameters=EditFileArguments,
        handler=_EditFileHandler(workspace, approval_callback, max_content_bytes),
    )


def create_workspace_registry(
    workspace: Workspace,
    *,
    allow_write: bool = False,
    allow_exec: bool = False,
    approval_callback: ApprovalCallback | None = None,
    command_limits: CommandLimits | None = None,
) -> ToolRegistry:
    """Create the standard registry with explicitly enabled side-effect tools."""
    registry = create_read_only_registry(workspace)
    if allow_write:
        registry.register(write_file_tool_spec(workspace, approval_callback))
        registry.register(edit_file_tool_spec(workspace, approval_callback))
    if allow_exec:
        registry.register(
            run_command_tool_spec(
                workspace,
                approval_callback,
                limits=command_limits,
            )
        )
    return registry


def _inspect_parent_path(
    workspace_root: Path,
    parts: tuple[str, ...],
) -> tuple[list[Path], list[_ParentFingerprint]]:
    missing_parents: list[Path] = []
    root_stat = _lstat(workspace_root)
    if root_stat is None or _is_reparse_point(root_stat) or not stat.S_ISDIR(root_stat.st_mode):
        raise _UnsafeWritePathError("Workspace root changed or is not a safe directory.")
    existing_parents = [_make_parent_fingerprint(workspace_root, root_stat)]
    current = workspace_root

    for part in parts[:-1]:
        current /= part
        path_stat = _lstat(current)
        if path_stat is None:
            missing_parents.append(current)
            continue
        if _is_reparse_point(path_stat):
            raise _UnsafeWritePathError(
                f"Write path crosses a symbolic link or reparse point: {part}"
            )
        if not stat.S_ISDIR(path_stat.st_mode):
            raise _UnsafeWritePathError(f"Write path parent is not a directory: {part}")
        existing_parents.append(_make_parent_fingerprint(current, path_stat))

    return missing_parents, existing_parents


def _make_parent_fingerprint(path: Path, path_stat: stat_result) -> _ParentFingerprint:
    return _ParentFingerprint(
        path=path,
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
        mode=path_stat.st_mode,
        modified_ns=path_stat.st_mtime_ns,
        changed_ns=path_stat.st_ctime_ns,
    )


def _matches_parent_fingerprint(
    fingerprint: _ParentFingerprint,
    path_stat: stat_result | None,
) -> bool:
    return (
        path_stat is not None
        and not _is_reparse_point(path_stat)
        and stat.S_ISDIR(path_stat.st_mode)
        and _make_parent_fingerprint(fingerprint.path, path_stat) == fingerprint
    )


def _same_parent_identities(
    first: tuple[_ParentFingerprint, ...],
    second: tuple[_ParentFingerprint, ...],
) -> bool:
    return len(first) == len(second) and all(
        left.path == right.path
        and left.device == right.device
        and left.inode == right.inode
        and left.mode == right.mode
        for left, right in zip(first, second, strict=True)
    )


def _count_overlapping_matches(text: str, fragment: str) -> int:
    """Count only far enough to distinguish zero, one, and multiple matches."""
    count = 0
    start = 0
    while count < 2:
        position = text.find(fragment, start)
        if position < 0:
            break
        count += 1
        start = position + 1
    return count


def _load_edit_source(
    workspace: Workspace,
    path: str,
    max_content_bytes: int,
) -> _EditSource:
    relative_path, parts = _normalize_write_path(path)
    missing_parents, parents = _inspect_parent_path(workspace.root, parts)
    if missing_parents:
        raise _TargetNotFoundError(relative_path)

    target = workspace.root.joinpath(*parts)
    before_stat = _lstat(target)
    if before_stat is None:
        raise _TargetNotFoundError(relative_path)
    _validate_edit_target(before_stat)

    content, text = _read_bounded_utf8(target, max_content_bytes)
    after_stat = _lstat(target)
    if after_stat is None:
        raise _EditConflictError("Edit target disappeared while it was being read.")
    _validate_edit_target(after_stat)
    if not _same_file_state(before_stat, after_stat) or len(content) != after_stat.st_size:
        raise _EditConflictError("Edit target changed while it was being read.")

    return _EditSource(
        relative_path=relative_path,
        parts=parts,
        target=target,
        text=text,
        parents=tuple(parents),
        fingerprint=_TargetFingerprint(
            device=after_stat.st_dev,
            inode=after_stat.st_ino,
            size=after_stat.st_size,
            modified_ns=after_stat.st_mtime_ns,
            mode=after_stat.st_mode,
            digest=hashlib.sha256(content).hexdigest(),
        ),
    )


def _validate_edit_target(path_stat: stat_result) -> None:
    if _is_reparse_point(path_stat):
        raise _UnsafeWritePathError("Edit target must not be a symbolic link or reparse point.")
    if not stat.S_ISREG(path_stat.st_mode):
        raise _UnsafeWritePathError("Edit target must be a regular file.")
    if path_stat.st_nlink > 1:
        raise _UnsafeWritePathError("Edit target with multiple hard links is not supported.")


def _same_file_state(first: stat_result, second: stat_result) -> bool:
    return (
        first.st_dev == second.st_dev
        and first.st_ino == second.st_ino
        and first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
        and first.st_mode == second.st_mode
    )


def _read_bounded_utf8(path: Path, max_content_bytes: int) -> tuple[bytes, str]:
    try:
        with path.open("rb") as file_handle:
            content = file_handle.read(max_content_bytes + 1)
    except OSError as exc:
        raise _FileReadError(f"Unable to read edit target: {path.name}") from exc

    if len(content) > max_content_bytes:
        raise _FileTooLargeError(f"Source file exceeds the {max_content_bytes}-byte UTF-8 limit.")
    if b"\x00" in content:
        raise _InvalidTextFileError("Edit target appears to be binary; UTF-8 text required.")
    try:
        return content, content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _InvalidTextFileError("Edit target is not valid UTF-8 text.") from exc


def _build_edit_preview(relative_path: str, original: str, edited: str) -> str:
    diff_lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        edited.splitlines(keepends=True),
        fromfile=f"a/{relative_path}",
        tofile=f"b/{relative_path}",
        lineterm="",
    )
    escaped_diff = "\n".join(json.dumps(line, ensure_ascii=False) for line in diff_lines)
    return "\n".join(
        (
            "Edit UTF-8 text file",
            f"Path: {relative_path}",
            "Unified diff (JSON-escaped lines):",
            escaped_diff,
        )
    )


def _normalize_write_path(path: str) -> tuple[str, tuple[str, ...]]:
    if "\x00" in path or any(ord(character) < 32 for character in path):
        raise _UnsafeWritePathError("Write path contains a control character.")

    windows_path = PureWindowsPath(path)
    if windows_path.is_absolute() or windows_path.drive or windows_path.root:
        raise _UnsafeWritePathError("Write path must be workspace-relative.")

    parts = windows_path.parts
    if not parts or any(part == ".." for part in parts):
        raise _UnsafeWritePathError("Write path must not be empty or escape the workspace.")

    for index, part in enumerate(parts):
        normalized_part = part.casefold()
        if normalized_part in _PROTECTED_DIRECTORY_NAMES:
            raise _UnsafeWritePathError(f"Write path enters protected directory: {part}")
        is_environment_path = normalized_part == ".env" or normalized_part.startswith(".env.")
        is_allowed_template = normalized_part == ".env.example" and index == len(parts) - 1
        if is_environment_path and not is_allowed_template:
            raise _UnsafeWritePathError("Writing real environment paths is not allowed.")
        if part.endswith((" ", ".")):
            raise _UnsafeWritePathError("Write path components must not end with a dot or space.")
        if any(character in '<>:"|?*' for character in part):
            raise _UnsafeWritePathError("Write path contains a Windows special character.")
        reserved_stem = part.split(".", 1)[0].upper()
        if reserved_stem in _WINDOWS_RESERVED_STEMS:
            raise _UnsafeWritePathError(f"Write path uses reserved device name: {part}")

    return "/".join(parts), parts


def _build_write_preview(
    relative_path: str,
    content: str,
    *,
    content_bytes: int,
    missing_parents: tuple[str, ...],
) -> str:
    directories = "none" if not missing_parents else ", ".join(missing_parents)
    encoded_content = json.dumps(content, ensure_ascii=False)
    return "\n".join(
        (
            "Create UTF-8 text file",
            f"Path: {relative_path}",
            f"Parent directories to create: {directories}",
            f"Content bytes: {content_bytes}",
            f"Content: {encoded_content}",
        )
    )


def _lstat(path: Path) -> stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _WritePathInspectionError(f"Unable to inspect write path: {path.name}") from exc


def _is_reparse_point(path_stat: stat_result) -> bool:
    return stat.S_ISLNK(path_stat.st_mode) or bool(
        getattr(path_stat, "st_reparse_tag", 0)
        or (getattr(path_stat, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT)
    )


def _cleanup_created_paths(
    target: Path | None,
    created_directories: list[Path],
) -> tuple[str, ...]:
    incomplete: list[str] = []
    if target is not None:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            incomplete.append(str(target))

    for directory in reversed(created_directories):
        try:
            directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            incomplete.append(str(directory))
    return tuple(incomplete)


def _cleanup_temporary_file(
    temporary_path: Path | None,
    workspace_root: Path,
) -> tuple[str, ...]:
    if temporary_path is None:
        return ()
    try:
        temporary_path.unlink()
    except FileNotFoundError:
        return ()
    except OSError:
        try:
            display_path = temporary_path.relative_to(workspace_root).as_posix()
        except ValueError:
            display_path = temporary_path.name
        return (display_path,)
    return ()


def _close_file_descriptor(descriptor: int | None) -> None:
    if descriptor is None:
        return
    with suppress(OSError):
        os.close(descriptor)
