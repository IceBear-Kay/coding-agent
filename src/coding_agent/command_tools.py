"""Approved local command execution contracts and runner."""

import codecs
import contextlib
import ctypes
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from os import stat_result
from pathlib import Path, PureWindowsPath
from typing import Protocol

if os.name == "nt":
    from ctypes import wintypes

from pydantic import BaseModel, ConfigDict, Field, field_validator

from coding_agent.approval import ApprovalCallback, ApprovalRequest, request_approval
from coding_agent.tools import JsonValue, ToolOutput, ToolSpec, Workspace

DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0
MAX_COMMAND_TIMEOUT_SECONDS = 60.0
DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES = 65_536
MAX_COMMAND_OUTPUT_LIMIT_BYTES = 1_048_576
MAX_COMMAND_ARGUMENTS = 128
MAX_COMMAND_ARGV_BYTES = 16_384
MAX_COMMAND_STDIN_BYTES = 65_536
_OUTPUT_READ_CHUNK_SIZE = 4_096
_PROCESS_POLL_INTERVAL_SECONDS = 0.02
_PROCESS_CLEANUP_WAIT_SECONDS = 1.0

_PYTHON_COMMAND_NAMES = frozenset({"python", "python3", "python3.12"})
_WINDOWS_SCRIPT_SUFFIXES = frozenset({".bat", ".cmd"})
_WINDOWS_REPARSE_POINT = 0x0400
_INHERITED_ENVIRONMENT_NAMES = (
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "WINDIR",
)


class RunCommandArguments(BaseModel):
    """Model-controlled arguments accepted by the run_command tool."""

    model_config = ConfigDict(extra="forbid")

    argv: list[str] = Field(min_length=1, max_length=MAX_COMMAND_ARGUMENTS)
    cwd: str = Field(default=".", min_length=1)
    stdin: str = ""

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, argv: list[str]) -> list[str]:
        if not argv[0]:
            raise ValueError("argv[0] must not be empty")
        if any("\x00" in argument for argument in argv):
            raise ValueError("argv must not contain NUL characters")
        if sum(len(argument.encode("utf-8")) for argument in argv) > MAX_COMMAND_ARGV_BYTES:
            raise ValueError(f"argv exceeds the {MAX_COMMAND_ARGV_BYTES}-byte UTF-8 limit")
        return argv

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, cwd: str) -> str:
        if "\x00" in cwd or any(ord(character) < 32 for character in cwd):
            raise ValueError("cwd must not contain control characters")
        return cwd

    @field_validator("stdin")
    @classmethod
    def validate_stdin(cls, stdin: str) -> str:
        if len(stdin.encode("utf-8")) > MAX_COMMAND_STDIN_BYTES:
            raise ValueError(f"stdin exceeds the {MAX_COMMAND_STDIN_BYTES}-byte UTF-8 limit")
        return stdin


@dataclass(frozen=True, slots=True)
class CommandLimits:
    """Application-owned limits that model arguments cannot override."""

    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
    output_limit_bytes: int = DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES

    def __post_init__(self) -> None:
        timeout = self.timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
            or timeout > MAX_COMMAND_TIMEOUT_SECONDS
        ):
            raise ValueError(
                f"timeout_seconds must be finite, greater than zero, and at most "
                f"{MAX_COMMAND_TIMEOUT_SECONDS}"
            )
        if (
            isinstance(self.output_limit_bytes, bool)
            or not isinstance(self.output_limit_bytes, int)
            or self.output_limit_bytes <= 0
            or self.output_limit_bytes > MAX_COMMAND_OUTPUT_LIMIT_BYTES
        ):
            raise ValueError(
                f"output_limit_bytes must be between 1 and {MAX_COMMAND_OUTPUT_LIMIT_BYTES}"
            )
        object.__setattr__(self, "timeout_seconds", float(timeout))


@dataclass(frozen=True, slots=True)
class _DirectoryFingerprint:
    path: Path
    device: int
    inode: int
    mode: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class _ExecutableFingerprint:
    path: Path
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True, slots=True)
class CommandPlan:
    """Exact command prepared before approval and rechecked before launch."""

    argv: tuple[str, ...]
    cwd: Path
    relative_cwd: str
    stdin: bytes
    limits: CommandLimits
    executable_fingerprint: _ExecutableFingerprint
    directory_fingerprints: tuple[_DirectoryFingerprint, ...]
    preview: str


class CommandRunner(Protocol):
    def run(self, plan: CommandPlan) -> ToolOutput:
        """Run one previously approved command plan."""


class _CommandPathError(ValueError):
    """The requested command working directory is unsafe or invalid."""


class _CommandPreparationError(ValueError):
    """The command cannot be resolved to a deterministic executable."""


class LocalCommandRunner:
    """Run short local commands without exposing the parent process environment."""

    def __init__(
        self,
        *,
        popen_factory: Callable[..., subprocess.Popen[bytes]] | None = None,
        environment: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._popen_factory = popen_factory or subprocess.Popen
        self._environment = _build_minimal_environment(environment)
        self._clock = clock

    def run(self, plan: CommandPlan) -> ToolOutput:
        """Execute a prepared short command with bounded, concurrent stream reads."""
        launch_started_at = self._clock()
        try:
            process = self._popen_factory(
                list(plan.argv),
                cwd=plan.cwd,
                env=self._environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                **_process_group_options(),
            )
        except OSError as exc:
            return _command_result(
                plan,
                status="launch_error",
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=self._clock() - launch_started_at,
                is_error=True,
                message=f"Unable to start command: {exc}",
            )

        started_at = self._clock()
        collector = _BoundedOutputCollector(plan.limits.output_limit_bytes)
        controller = _ProcessController(process)
        reader_threads = [
            threading.Thread(
                target=_read_output_stream,
                args=(process.stdout, "stdout", collector, controller),
                name="coding-agent-stdout-reader",
                daemon=True,
            ),
            threading.Thread(
                target=_read_output_stream,
                args=(process.stderr, "stderr", collector, controller),
                name="coding-agent-stderr-reader",
                daemon=True,
            ),
        ]
        stdin_thread = threading.Thread(
            target=_write_stdin,
            args=(process, plan.stdin),
            name="coding-agent-stdin-writer",
            daemon=True,
        )
        for thread in (*reader_threads, stdin_thread):
            thread.start()

        timed_out = False
        try:
            timed_out = _wait_for_process(
                process,
                deadline=started_at + plan.limits.timeout_seconds,
                clock=self._clock,
                controller=controller,
            )
            if timed_out:
                _close_process_streams(process)
                _wait_for_exit(process, _PROCESS_CLEANUP_WAIT_SECONDS)
            elif not _join_threads_until(
                reader_threads + [stdin_thread],
                started_at + plan.limits.timeout_seconds,
                self._clock,
            ):
                timed_out = True
                controller.terminate("timeout")
                _close_process_streams(process)
                _wait_for_exit(process, _PROCESS_CLEANUP_WAIT_SECONDS)
        except KeyboardInterrupt:
            controller.terminate("interrupted")
            _close_process_streams(process)
            _wait_for_exit(process, _PROCESS_CLEANUP_WAIT_SECONDS)
            _join_threads_bounded(reader_threads + [stdin_thread], _PROCESS_CLEANUP_WAIT_SECONDS)
            controller.close()
            raise

        if timed_out:
            _join_threads_bounded(reader_threads + [stdin_thread], _PROCESS_CLEANUP_WAIT_SECONDS)
        else:
            _close_process_streams(process)
        controller.close()

        exit_code = process.returncode
        truncated = collector.truncated
        if truncated:
            status = "output_limit"
            message = "Combined stdout and stderr output exceeded the configured limit."
        elif timed_out:
            status = "timeout"
            message = "Command exceeded its configured timeout."
        elif collector.io_error is not None:
            status = "io_error"
            message = collector.io_error
        else:
            status = "completed" if exit_code == 0 else "failed"
            message = None
        if controller.cleanup_error is not None:
            cleanup_message = f"Process cleanup warning: {controller.cleanup_error}"
            message = f"{message} {cleanup_message}" if message else cleanup_message
            status = "cleanup_failed"
        return _command_result(
            plan,
            status=status,
            exit_code=exit_code,
            stdout=_decode_output(collector.data("stdout"), truncated=truncated),
            stderr=_decode_output(collector.data("stderr"), truncated=truncated),
            duration_seconds=self._clock() - started_at,
            is_error=status != "completed",
            message=message,
            truncated=truncated,
        )


class _ProcessController:
    """Coordinate one best-effort process-tree termination across threads."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self._lock = threading.Lock()
        self._windows_job = _create_windows_job(process)
        self.reason: str | None = None
        self.cleanup_error: str | None = None

    def terminate(self, reason: str) -> None:
        with self._lock:
            if self.reason is not None:
                return
            self.reason = reason
        try:
            if self._windows_job is not None:
                self._windows_job.terminate()
            else:
                _terminate_process_tree(self._process)
        except (OSError, subprocess.SubprocessError) as exc:
            self.cleanup_error = str(exc)

    def close(self) -> None:
        if self._windows_job is None:
            return
        try:
            self._windows_job.close()
        except OSError as exc:
            self.cleanup_error = str(exc)


if os.name == "nt":

    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobObjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


class _WindowsJob:
    """Own a Windows Job Object configured to kill its process tree on close."""

    def __init__(self, kernel32: object, handle: int) -> None:
        self._kernel32 = kernel32
        self._handle = handle

    def terminate(self) -> None:
        if self._handle and not self._kernel32.TerminateJobObject(self._handle, 1):  # type: ignore[attr-defined]
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._handle:
            handle, self._handle = self._handle, 0
            if not self._kernel32.CloseHandle(handle):  # type: ignore[attr-defined]
                raise ctypes.WinError(ctypes.get_last_error())


def _create_windows_job(process: subprocess.Popen[bytes]) -> _WindowsJob | None:
    if os.name != "nt" or not hasattr(process, "_handle"):
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return None

    job = _WindowsJob(kernel32, handle)
    information = _JobObjectExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    configured = kernel32.SetInformationJobObject(
        handle,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    assigned = configured and kernel32.AssignProcessToJobObject(handle, process._handle)
    if assigned:
        return job
    with contextlib.suppress(OSError):
        job.close()
    return None


class _BoundedOutputCollector:
    """Collect stdout and stderr under one shared byte budget."""

    def __init__(self, limit: int) -> None:
        self._remaining = limit
        self._buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        self._condition = threading.Condition()
        self.truncated = False
        self.io_error: str | None = None

    def reserve(self, requested: int) -> int:
        with self._condition:
            while self._remaining <= 0 and not self.truncated:
                self._condition.wait()
            if self.truncated:
                return 0
            reserved = min(requested, self._remaining)
            self._remaining -= reserved
            return reserved

    def record(self, stream_name: str, data: bytes, reserved: int) -> None:
        with self._condition:
            if len(data) < reserved:
                self._remaining += reserved - len(data)
            elif self._remaining == 0:
                self.truncated = True
            self._buffers[stream_name].extend(data[:reserved])
            self._condition.notify_all()

    def data(self, stream_name: str) -> bytes:
        with self._condition:
            return bytes(self._buffers[stream_name])

    def fail(self, message: str) -> None:
        with self._condition:
            if self.io_error is None:
                self.io_error = message
            self._condition.notify_all()


def _read_output_stream(
    stream: object,
    stream_name: str,
    collector: _BoundedOutputCollector,
    controller: _ProcessController,
) -> None:
    if stream is None:
        return
    while True:
        reserved = collector.reserve(_OUTPUT_READ_CHUNK_SIZE)
        if reserved == 0:
            controller.terminate("output_limit")
            return
        try:
            data = stream.read(reserved)  # type: ignore[union-attr]
        except (OSError, ValueError) as exc:
            collector.record(stream_name, b"", reserved)
            collector.fail(f"Unable to read {stream_name}: {exc}")
            controller.terminate("io_error")
            return
        if not data:
            collector.record(stream_name, b"", reserved)
            return
        collector.record(stream_name, data, reserved)
        if collector.truncated:
            controller.terminate("output_limit")
            return


def _write_stdin(process: subprocess.Popen[bytes], data: bytes) -> None:
    stream = process.stdin
    if stream is None:
        return
    try:
        if data:
            offset = 0
            while offset < len(data):
                written = stream.write(data[offset:])
                if written is None or written <= 0:
                    raise OSError("stdin pipe did not accept input")
                offset += written
            stream.flush()
    except (BrokenPipeError, OSError, ValueError):
        pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            stream.close()


def _process_group_options() -> dict[str, int | bool]:
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creation_flags} if creation_flags else {}
    return {"start_new_session": True}


def _process_poll(process: subprocess.Popen[bytes]) -> int | None:
    poll = getattr(process, "poll", None)
    return poll() if poll is not None else getattr(process, "returncode", None)


def _wait_for_process(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
    clock: Callable[[], float],
    controller: _ProcessController,
) -> bool:
    """Wait until the process exits or terminate it when the deadline expires."""
    while _process_poll(process) is None:
        remaining = deadline - clock()
        if remaining <= 0:
            controller.terminate("timeout")
            return True
        try:
            process.wait(timeout=min(remaining, _PROCESS_POLL_INTERVAL_SECONDS))
        except subprocess.TimeoutExpired:
            continue
    return False


def _wait_for_exit(process: subprocess.Popen[bytes], timeout: float) -> bool:
    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError, TypeError, KeyboardInterrupt):
        return False
    return True


def _join_threads_until(
    threads: list[threading.Thread],
    deadline: float,
    clock: Callable[[], float],
) -> bool:
    for thread in threads:
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        thread.join(timeout=remaining)
        if thread.is_alive():
            return False
    return True


def _join_threads_bounded(threads: list[threading.Thread], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))


def _close_process_streams(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            with contextlib.suppress(OSError, ValueError):
                stream.close()


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    pid = getattr(process, "pid", None)
    if pid is not None and os.name == "nt":
        taskkill = _resolve_taskkill()
        taskkill_error: Exception | None = None
        if taskkill is not None:
            try:
                completed = subprocess.run(
                    [taskkill, "/PID", str(pid), "/T", "/F"],
                    check=False,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=_PROCESS_CLEANUP_WAIT_SECONDS,
                )
                if completed.returncode == 0:
                    return
            except (OSError, subprocess.SubprocessError) as exc:
                taskkill_error = exc
        try:
            process.kill()
        except (OSError, ProcessLookupError) as exc:
            if taskkill_error is not None:
                raise taskkill_error from exc
        return

    if pid is not None and os.name != "nt":
        try:
            os.killpg(pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    with contextlib.suppress(OSError, ProcessLookupError):
        process.kill()


def _resolve_taskkill() -> str | None:
    system_root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR")
    if system_root:
        candidate = Path(system_root) / "System32" / "taskkill.exe"
        if candidate.is_file():
            return str(candidate)
    resolved = shutil.which("taskkill.exe")
    return str(Path(resolved).resolve()) if resolved else None


def _decode_output(data: bytes, *, truncated: bool) -> str:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = decoder.decode(data, final=not truncated)
    if not truncated:
        text += decoder.decode(b"", final=True)
    return text


class _RunCommandHandler:
    def __init__(
        self,
        workspace: Workspace,
        approval_callback: ApprovalCallback | None,
        limits: CommandLimits,
        runner: CommandRunner,
    ) -> None:
        self._workspace = workspace
        self._approval_callback = approval_callback
        self._limits = limits
        self._runner = runner

    def __call__(self, arguments: RunCommandArguments) -> ToolOutput:
        try:
            plan = _prepare_command(self._workspace, arguments, self._limits)
        except _CommandPathError as exc:
            return _command_error("invalid_path", arguments, str(exc))
        except _CommandPreparationError as exc:
            return _command_error("launch_error", arguments, str(exc))

        request = ApprovalRequest(operation="run_command", preview=plan.preview)
        try:
            approved = request_approval(request, self._approval_callback)
        except Exception as exc:
            return _plan_error(
                "approval_failed",
                plan,
                f"Approval could not be obtained: {exc}",
            )
        if not approved:
            return _plan_error(
                "denied",
                plan,
                "Operation was not approved and no process was started.",
            )

        try:
            verified_plan = _prepare_command(self._workspace, arguments, self._limits)
        except (_CommandPathError, _CommandPreparationError) as exc:
            return _plan_error(
                "conflict",
                plan,
                f"Command state changed after approval: {exc}",
            )
        if verified_plan != plan:
            return _plan_error(
                "conflict",
                plan,
                "Command or working directory changed after approval.",
            )
        return self._runner.run(plan)


def run_command_tool_spec(
    workspace: Workspace,
    approval_callback: ApprovalCallback | None = None,
    *,
    limits: CommandLimits | None = None,
    runner: CommandRunner | None = None,
) -> ToolSpec[RunCommandArguments]:
    """Build the approved run_command tool specification."""
    selected_limits = limits or CommandLimits()
    selected_runner = runner or LocalCommandRunner()
    return ToolSpec(
        name="run_command",
        description="Run an approved local command with structured argv inside the workspace.",
        parameters=RunCommandArguments,
        handler=_RunCommandHandler(
            workspace,
            approval_callback,
            selected_limits,
            selected_runner,
        ),
    )


def _prepare_command(
    workspace: Workspace,
    arguments: RunCommandArguments,
    limits: CommandLimits,
) -> CommandPlan:
    cwd, relative_cwd, fingerprints = _prepare_cwd(workspace, arguments.cwd)
    argv, executable_fingerprint = _resolve_argv(arguments.argv, cwd, workspace.root)
    stdin = arguments.stdin.encode("utf-8")
    preview = _build_command_preview(argv, relative_cwd, arguments.stdin, stdin, limits)
    return CommandPlan(
        argv=argv,
        cwd=cwd,
        relative_cwd=relative_cwd,
        stdin=stdin,
        limits=limits,
        executable_fingerprint=executable_fingerprint,
        directory_fingerprints=fingerprints,
        preview=preview,
    )


def _prepare_cwd(
    workspace: Workspace,
    cwd: str,
) -> tuple[Path, str, tuple[_DirectoryFingerprint, ...]]:
    windows_path = PureWindowsPath(cwd)
    native_path = Path(cwd)
    if (
        windows_path.is_absolute()
        or windows_path.anchor
        or native_path.is_absolute()
        or native_path.anchor
        or any(part == ".." for part in windows_path.parts)
    ):
        raise _CommandPathError("Command cwd must be a workspace-relative directory.")

    current = workspace.root
    fingerprints: list[_DirectoryFingerprint] = []
    root_stat = _lstat_directory(current)
    fingerprints.append(_make_directory_fingerprint(current, root_stat))

    for part in native_path.parts:
        if part == ".":
            continue
        current /= part
        path_stat = _lstat_directory(current)
        fingerprints.append(_make_directory_fingerprint(current, path_stat))

    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(workspace.root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise _CommandPathError("Command cwd escapes the workspace.") from exc

    relative_cwd = (
        "." if resolved == workspace.root else resolved.relative_to(workspace.root).as_posix()
    )
    return resolved, relative_cwd, tuple(fingerprints)


def _lstat_directory(path: Path) -> stat_result:
    try:
        path_stat = path.lstat()
    except FileNotFoundError as exc:
        raise _CommandPathError(f"Command cwd does not exist: {path.name}") from exc
    except OSError as exc:
        raise _CommandPathError(f"Unable to inspect command cwd: {path.name}") from exc
    if _is_reparse_point(path_stat):
        raise _CommandPathError("Command cwd must not cross a symbolic link or reparse point.")
    if not stat.S_ISDIR(path_stat.st_mode):
        raise _CommandPathError(f"Command cwd is not a directory: {path.name}")
    return path_stat


def _make_directory_fingerprint(
    path: Path,
    path_stat: stat_result,
) -> _DirectoryFingerprint:
    return _DirectoryFingerprint(
        path=path,
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
        mode=path_stat.st_mode,
        modified_ns=path_stat.st_mtime_ns,
        changed_ns=path_stat.st_ctime_ns,
    )


def _is_reparse_point(path_stat: stat_result) -> bool:
    return stat.S_ISLNK(path_stat.st_mode) or bool(
        getattr(path_stat, "st_reparse_tag", 0)
        or (getattr(path_stat, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT)
    )


def _resolve_argv(
    argv: list[str],
    cwd: Path,
    workspace_root: Path,
) -> tuple[tuple[str, ...], _ExecutableFingerprint]:
    executable = argv[0]
    if executable.casefold() in _PYTHON_COMMAND_NAMES:
        resolved_executable = str(Path(sys.executable).resolve(strict=True))
    elif _has_path_separator(executable):
        executable_path = Path(executable)
        candidate = executable_path if executable_path.is_absolute() else cwd / executable_path
        try:
            resolved_path = candidate.resolve(strict=True)
        except OSError as exc:
            raise _CommandPreparationError(
                f"Command executable was not found: {executable}"
            ) from exc
        if not executable_path.is_absolute():
            try:
                resolved_path.relative_to(workspace_root)
            except ValueError as exc:
                raise _CommandPreparationError(
                    "Relative command executable must stay inside the workspace."
                ) from exc
        resolved_executable = str(resolved_path)
    else:
        resolved = shutil.which(executable)
        if resolved is None:
            raise _CommandPreparationError(f"Command executable was not found: {executable}")
        resolved_executable = str(Path(resolved).resolve(strict=True))

    if Path(resolved_executable).suffix.casefold() in _WINDOWS_SCRIPT_SUFFIXES:
        raise _CommandPreparationError("Windows batch scripts are not supported.")
    if not Path(resolved_executable).is_file():
        raise _CommandPreparationError("Command executable must resolve to a file.")
    executable_path = Path(resolved_executable)
    try:
        executable_stat = executable_path.stat()
    except OSError as exc:
        raise _CommandPreparationError("Command executable could not be inspected.") from exc
    fingerprint = _ExecutableFingerprint(
        path=executable_path,
        device=executable_stat.st_dev,
        inode=executable_stat.st_ino,
        mode=executable_stat.st_mode,
        size=executable_stat.st_size,
        modified_ns=executable_stat.st_mtime_ns,
        changed_ns=executable_stat.st_ctime_ns,
    )
    return (resolved_executable, *argv[1:]), fingerprint


def _has_path_separator(value: str) -> bool:
    return "/" in value or "\\" in value


def _build_minimal_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    parent = os.environ if source is None else source
    environment = {
        name: parent[name]
        for name in _INHERITED_ENVIRONMENT_NAMES
        if name in parent and parent[name]
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    return environment


def _build_command_preview(
    argv: tuple[str, ...],
    relative_cwd: str,
    stdin_text: str,
    stdin_bytes: bytes,
    limits: CommandLimits,
) -> str:
    return "\n".join(
        (
            "Run local command",
            f"Argv: {json.dumps(list(argv), ensure_ascii=False)}",
            f"Cwd: {relative_cwd}",
            f"Stdin bytes: {len(stdin_bytes)}",
            f"Stdin: {json.dumps(stdin_text, ensure_ascii=False)}",
            f"Timeout seconds: {limits.timeout_seconds}",
            f"Combined output limit bytes: {limits.output_limit_bytes}",
        )
    )


def _command_result(
    plan: CommandPlan,
    *,
    status: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    duration_seconds: float,
    is_error: bool,
    message: str | None = None,
    truncated: bool = False,
) -> ToolOutput:
    details: dict[str, JsonValue] = {
        "argv": list(plan.argv),
        "cwd": plan.relative_cwd,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration_seconds": round(max(0.0, duration_seconds), 6),
        "truncated": truncated,
    }
    if message is not None:
        details["message"] = message
    return ToolOutput(status=status, details=details, is_error=is_error)


def _plan_error(status: str, plan: CommandPlan, message: str) -> ToolOutput:
    return ToolOutput(
        status=status,
        details={
            "argv": list(plan.argv),
            "cwd": plan.relative_cwd,
            "message": message,
        },
        is_error=True,
    )


def _command_error(
    status: str,
    arguments: RunCommandArguments,
    message: str,
) -> ToolOutput:
    return ToolOutput(
        status=status,
        details={
            "argv": list(arguments.argv),
            "cwd": arguments.cwd,
            "message": message,
        },
        is_error=True,
    )
