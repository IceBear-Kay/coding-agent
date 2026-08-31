import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from coding_agent.approval import ApprovalRequest
from coding_agent.command_tools import (
    DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES,
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    MAX_COMMAND_ARGUMENTS,
    MAX_COMMAND_ARGV_BYTES,
    MAX_COMMAND_OUTPUT_LIMIT_BYTES,
    MAX_COMMAND_STDIN_BYTES,
    MAX_COMMAND_TIMEOUT_SECONDS,
    CommandLimits,
    CommandPlan,
    LocalCommandRunner,
    RunCommandArguments,
    _prepare_command,
    run_command_tool_spec,
)
from coding_agent.models import ToolCall
from coding_agent.tools import ToolDispatcher, ToolOutput, ToolRegistry, Workspace


def dispatch_command(
    workspace: Workspace,
    arguments: dict[str, object],
    approval_callback=None,
    *,
    limits: CommandLimits | None = None,
    runner=None,
):
    spec = run_command_tool_spec(
        workspace,
        approval_callback,
        limits=limits,
        runner=runner,
    )
    return ToolDispatcher(ToolRegistry([spec])).dispatch(
        ToolCall(id="call_command", name="run_command", arguments=arguments)
    )


def make_command_plan(tmp_path: Path, limits: CommandLimits) -> CommandPlan:
    return _prepare_command(
        Workspace(tmp_path),
        RunCommandArguments(argv=["python"]),
        limits,
    )


def test_run_command_schema_only_exposes_model_controlled_fields(tmp_path: Path) -> None:
    schema = run_command_tool_spec(Workspace(tmp_path)).parameters_schema

    assert set(schema["properties"]) == {"argv", "cwd", "stdin"}
    assert schema["required"] == ["argv"]
    assert schema["properties"]["argv"]["minItems"] == 1
    assert schema["properties"]["argv"]["maxItems"] == MAX_COMMAND_ARGUMENTS
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        {"argv": []},
        {"argv": [""]},
        {"argv": ["python", "bad\x00argument"]},
        {"argv": ["python", *("x" for _ in range(MAX_COMMAND_ARGUMENTS))]},
        {"argv": ["python", "x" * MAX_COMMAND_ARGV_BYTES]},
        {"argv": ["python"], "cwd": "bad\x00cwd"},
        {"argv": ["python"], "stdin": "你" * (MAX_COMMAND_STDIN_BYTES // 3 + 1)},
        {"argv": ["python"], "env": {"TOKEN": "value"}},
        {"argv": ["python"], "shell": True},
        {"argv": ["python"], "approved": True},
        {"argv": ["python"], "timeout": 999},
    ],
)
def test_run_command_rejects_invalid_model_arguments_before_approval(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    result = dispatch_command(Workspace(tmp_path), arguments, approve)

    assert result.is_error is True
    assert "Invalid arguments" in result.content
    assert approval_calls == 0


def test_run_command_accepts_argument_and_stdin_byte_boundaries() -> None:
    arguments = RunCommandArguments(
        argv=["p", "x" * (MAX_COMMAND_ARGV_BYTES - 1)],
        stdin="a" * MAX_COMMAND_STDIN_BYTES,
    )

    assert sum(len(value.encode()) for value in arguments.argv) == MAX_COMMAND_ARGV_BYTES
    assert len(arguments.stdin.encode()) == MAX_COMMAND_STDIN_BYTES


def test_command_limits_use_bounded_defaults() -> None:
    limits = CommandLimits()

    assert limits.timeout_seconds == DEFAULT_COMMAND_TIMEOUT_SECONDS
    assert limits.output_limit_bytes == DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES


@pytest.mark.parametrize(
    "timeout",
    [True, 0, -1, math.nan, math.inf, -math.inf, MAX_COMMAND_TIMEOUT_SECONDS + 0.1, "10"],
)
def test_command_limits_reject_invalid_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        CommandLimits(timeout_seconds=timeout)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "output_limit",
    [True, 0, -1, MAX_COMMAND_OUTPUT_LIMIT_BYTES + 1, 1.5, "1024"],
)
def test_command_limits_reject_invalid_output_limit(output_limit: object) -> None:
    with pytest.raises(ValueError, match="output_limit_bytes"):
        CommandLimits(output_limit_bytes=output_limit)  # type: ignore[arg-type]


class RecordingRunner:
    def __init__(self) -> None:
        self.plans: list[CommandPlan] = []

    def run(self, plan: CommandPlan) -> ToolOutput:
        self.plans.append(plan)
        return ToolOutput(status="completed", details={"exit_code": 0})


def test_run_command_defaults_to_denied_without_starting_runner(tmp_path: Path) -> None:
    runner = RecordingRunner()

    result = dispatch_command(
        Workspace(tmp_path),
        {"argv": ["python", "-c", "print('not started')"]},
        runner=runner,
    )

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "denied"
    assert runner.plans == []


def test_run_command_approval_previews_exact_resolved_plan(tmp_path: Path) -> None:
    work_dir = tmp_path / "work dir"
    work_dir.mkdir()
    requests: list[ApprovalRequest] = []
    runner = RecordingRunner()
    limits = CommandLimits(timeout_seconds=4.5, output_limit_bytes=1234)

    def approve(request: ApprovalRequest) -> bool:
        requests.append(request)
        return True

    result = dispatch_command(
        Workspace(tmp_path),
        {
            "argv": ["python", "script.py", "a&b"],
            "cwd": "work dir",
            "stdin": "7 5\n",
        },
        approve,
        limits=limits,
        runner=runner,
    )

    assert result.is_error is False
    assert len(requests) == 1
    assert requests[0].operation == "run_command"
    assert f"Argv: {json.dumps([str(Path(sys.executable).resolve()), 'script.py', 'a&b'])}" in (
        requests[0].preview
    )
    assert "Cwd: work dir" in requests[0].preview
    assert "Stdin bytes: 4" in requests[0].preview
    assert 'Stdin: "7 5\\n"' in requests[0].preview
    assert "Timeout seconds: 4.5" in requests[0].preview
    assert "Combined output limit bytes: 1234" in requests[0].preview
    assert runner.plans[0].argv[0] == str(Path(sys.executable).resolve())
    assert runner.plans[0].stdin == b"7 5\n"


def test_run_command_returns_approval_failure_without_starting_runner(tmp_path: Path) -> None:
    runner = RecordingRunner()

    def fail_approval(_: ApprovalRequest) -> bool:
        raise RuntimeError("approval unavailable")

    result = dispatch_command(
        Workspace(tmp_path),
        {"argv": ["python"]},
        fail_approval,
        runner=runner,
    )

    payload = json.loads(result.content)
    assert payload["status"] == "approval_failed"
    assert runner.plans == []


def test_run_command_detects_cwd_replacement_after_approval(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    runner = RecordingRunner()

    def replace_cwd(_: ApprovalRequest) -> bool:
        work_dir.rename(tmp_path / "old-work")
        work_dir.mkdir()
        return True

    result = dispatch_command(
        Workspace(tmp_path),
        {"argv": ["python"], "cwd": "work"},
        replace_cwd,
        runner=runner,
    )

    assert json.loads(result.content)["status"] == "conflict"
    assert runner.plans == []


def test_run_command_detects_executable_replacement_after_approval(tmp_path: Path) -> None:
    executable = tmp_path / f"python-copy{Path(sys.executable).suffix}"
    shutil.copy2(sys.executable, executable)
    runner = RecordingRunner()

    def replace_executable(_: ApprovalRequest) -> bool:
        executable.write_bytes(b"replaced")
        return True

    result = dispatch_command(
        Workspace(tmp_path),
        {"argv": [f"./{executable.name}"]},
        replace_executable,
        runner=runner,
    )

    assert json.loads(result.content)["status"] == "conflict"
    assert runner.plans == []


def test_run_command_rejects_missing_executable_before_approval(tmp_path: Path) -> None:
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    result = dispatch_command(
        Workspace(tmp_path),
        {"argv": ["definitely-not-a-real-coding-agent-command"]},
        approve,
    )

    assert json.loads(result.content)["status"] == "launch_error"
    assert approval_calls == 0


@pytest.mark.parametrize("cwd", ["../outside", "missing", "file.txt"])
def test_run_command_rejects_invalid_cwd_before_approval(tmp_path: Path, cwd: str) -> None:
    (tmp_path / "file.txt").write_text("not a directory", encoding="utf-8")
    approval_calls = 0

    def approve(_: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    result = dispatch_command(Workspace(tmp_path), {"argv": ["python"], "cwd": cwd}, approve)

    assert result.is_error is True
    assert json.loads(result.content)["status"] == "invalid_path"
    assert approval_calls == 0


def test_run_command_executes_real_python_with_stdin_and_both_output_streams(
    tmp_path: Path,
) -> None:
    result = dispatch_command(
        Workspace(tmp_path),
        {
            "argv": [
                "python",
                "-c",
                "import sys; data=sys.stdin.read(); print(data.upper(), end=''); "
                "print('warning', file=sys.stderr)",
            ],
            "stdin": "hello\n",
        },
        lambda _: True,
    )

    payload = json.loads(result.content)
    assert result.is_error is False
    assert payload["status"] == "completed"
    assert payload["argv"][0] == str(Path(sys.executable).resolve())
    assert payload["cwd"] == "."
    assert payload["exit_code"] == 0
    assert payload["stdout"] == f"HELLO{os.linesep}"
    assert payload["stderr"] == f"warning{os.linesep}"
    assert payload["duration_seconds"] >= 0
    assert payload["truncated"] is False


def test_run_command_closes_empty_stdin_and_child_observes_eof(tmp_path: Path) -> None:
    result = dispatch_command(
        Workspace(tmp_path),
        {
            "argv": [
                "python",
                "-c",
                "import sys; print(repr(sys.stdin.read()))",
            ]
        },
        lambda _: True,
    )

    payload = json.loads(result.content)
    assert result.is_error is False
    assert payload["stdout"] == f"''{os.linesep}"


def test_run_command_preserves_chinese_utf8_output(tmp_path: Path) -> None:
    result = dispatch_command(
        Workspace(tmp_path),
        {
            "argv": ["python", "-c", """print('你好，世界')"""],
        },
        lambda _: True,
    )

    payload = json.loads(result.content)
    assert result.is_error is False
    assert payload["stdout"] == f"你好，世界{os.linesep}"


def test_run_command_stops_when_combined_output_reaches_limit(tmp_path: Path) -> None:
    result = dispatch_command(
        Workspace(tmp_path),
        {
            "argv": [
                "python",
                "-c",
                "import sys; sys.stdout.write('甲' * 10000); "
                "sys.stderr.write('乙' * 10000); sys.stdout.flush(); sys.stderr.flush()",
            ]
        },
        lambda _: True,
        limits=CommandLimits(output_limit_bytes=256),
    )

    payload = json.loads(result.content)
    output_bytes = len(payload["stdout"].encode()) + len(payload["stderr"].encode())
    assert result.is_error is True
    assert payload["status"] == "output_limit"
    assert payload["truncated"] is True
    assert output_bytes <= 256
    assert "message" in payload
    assert "\ufffd" not in payload["stdout"] + payload["stderr"]


def test_run_command_does_not_read_beyond_shared_output_budget(tmp_path: Path) -> None:
    class CountingStream:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self.read_bytes = 0
            self.read_sizes: list[int] = []

        def read(self, size: int) -> bytes:
            self.read_sizes.append(size)
            chunk, self._data = self._data[:size], self._data[size:]
            self.read_bytes += len(chunk)
            return chunk

        def close(self) -> None:
            pass

    class InputStream:
        def write(self, data: bytes) -> int:
            return len(data)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = CountingStream(b"a" * 10_000)
            self.stderr = CountingStream(b"b" * 10_000)
            self.stdin = InputStream()
            self.returncode = 0
            self.killed = False

        def wait(self) -> int:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = FakeProcess()
    plan = make_command_plan(tmp_path, CommandLimits(output_limit_bytes=128))

    result = LocalCommandRunner(popen_factory=lambda *args, **kwargs: process).run(plan)
    payload = json.loads(result.to_json())

    assert payload["status"] == "output_limit"
    assert process.killed is True
    assert process.stdout.read_bytes + process.stderr.read_bytes <= 128
    assert all(size <= 128 for size in process.stdout.read_sizes + process.stderr.read_sizes)


def test_run_command_times_out_and_terminates_process(tmp_path: Path) -> None:
    started_at = time.monotonic()
    result = dispatch_command(
        Workspace(tmp_path),
        {"argv": ["python", "-c", "import time; time.sleep(30)"]},
        lambda _: True,
        limits=CommandLimits(timeout_seconds=0.1),
    )
    elapsed = time.monotonic() - started_at

    payload = json.loads(result.content)
    assert result.is_error is True
    assert payload["status"] == "timeout"
    assert payload["truncated"] is False
    assert elapsed < 5


def test_run_command_timeout_cleans_child_that_keeps_output_pipe_open(tmp_path: Path) -> None:
    result = dispatch_command(
        Workspace(tmp_path),
        {
            "argv": [
                "python",
                "-c",
                "import subprocess, sys, time; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)']); "
                "time.sleep(30)",
            ]
        },
        lambda _: True,
        limits=CommandLimits(timeout_seconds=0.1),
    )

    payload = json.loads(result.content)
    assert result.is_error is True
    assert payload["status"] == "timeout"


def test_run_command_cleans_child_after_parent_exits_with_pipe_open(tmp_path: Path) -> None:
    marker = tmp_path / "orphan-marker.txt"
    child_code = (
        "import pathlib, time; time.sleep(0.8); "
        "pathlib.Path('orphan-marker.txt').write_text('alive', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print(child.pid, flush=True); time.sleep(0.1)"
    )

    result = dispatch_command(
        Workspace(tmp_path),
        {"argv": ["python", "-c", parent_code]},
        lambda _: True,
        limits=CommandLimits(timeout_seconds=0.4),
    )
    time.sleep(0.8)

    payload = json.loads(result.content)
    assert payload["status"] == "timeout"
    assert payload["stdout"].strip().isdigit()
    assert not marker.exists()


def test_run_command_propagates_keyboard_interrupt_after_cleanup(tmp_path: Path) -> None:
    class Stream:
        def __init__(self) -> None:
            self.closed = False

        def read(self, size: int) -> bytes:
            return b""

        def close(self) -> None:
            self.closed = True

    class InputStream(Stream):
        def write(self, data: bytes) -> int:
            return len(data)

        def flush(self) -> None:
            pass

    class InterruptingProcess:
        def __init__(self) -> None:
            self.stdout = Stream()
            self.stderr = Stream()
            self.stdin = InputStream()
            self.returncode = None
            self.killed = False

        def poll(self) -> None:
            return None

        def wait(self, **kwargs: object) -> int:
            raise KeyboardInterrupt

        def kill(self) -> None:
            self.killed = True
            self.returncode = -2

    process = InterruptingProcess()
    plan = make_command_plan(tmp_path, CommandLimits(timeout_seconds=10))

    with pytest.raises(KeyboardInterrupt):
        LocalCommandRunner(popen_factory=lambda *args, **kwargs: process).run(plan)

    assert process.killed is True
    assert process.stdin.closed is True
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_run_command_returns_structured_launch_error(tmp_path: Path) -> None:
    def deny_launch(*args: object, **kwargs: object) -> None:
        raise PermissionError("launch denied")

    runner = LocalCommandRunner(popen_factory=deny_launch)
    result = dispatch_command(
        Workspace(tmp_path),
        {"argv": ["python"]},
        lambda _: True,
        runner=runner,
    )

    payload = json.loads(result.content)
    assert result.is_error is True
    assert payload["status"] == "launch_error"
    assert payload["exit_code"] is None
    assert "launch denied" in payload["message"]


def test_run_command_returns_io_error_and_stops_process(tmp_path: Path) -> None:
    class FailingOutput:
        def read(self, size: int) -> bytes:
            raise PermissionError("read denied")

        def close(self) -> None:
            pass

    class EmptyOutput:
        def read(self, size: int) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class InputStream:
        def close(self) -> None:
            pass

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FailingOutput()
            self.stderr = EmptyOutput()
            self.stdin = InputStream()
            self.returncode = None
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    process = FakeProcess()
    result = LocalCommandRunner(popen_factory=lambda *args, **kwargs: process).run(
        make_command_plan(tmp_path, CommandLimits())
    )
    payload = json.loads(result.to_json())

    assert payload["status"] == "io_error"
    assert "read denied" in payload["message"]
    assert process.killed is True


def test_run_command_returns_structured_nonzero_exit(tmp_path: Path) -> None:
    result = dispatch_command(
        Workspace(tmp_path),
        {
            "argv": [
                "python",
                "-c",
                "import sys; print('failed', file=sys.stderr); raise SystemExit(7)",
            ]
        },
        lambda _: True,
    )

    payload = json.loads(result.content)
    assert result.is_error is True
    assert payload["status"] == "failed"
    assert payload["exit_code"] == 7
    assert payload["stderr"] == f"failed{os.linesep}"
    assert payload["truncated"] is False


def test_run_command_keeps_shell_metacharacters_as_literal_arguments(tmp_path: Path) -> None:
    literal_arguments = ["a&b", "$(not-a-command)", "value with spaces", ">output.txt"]
    result = dispatch_command(
        Workspace(tmp_path),
        {
            "argv": [
                "python",
                "-c",
                "import json, sys; print(json.dumps(sys.argv[1:]))",
                *literal_arguments,
            ]
        },
        lambda _: True,
    )

    payload = json.loads(result.content)
    assert result.is_error is False
    assert json.loads(payload["stdout"]) == literal_arguments
    assert not (tmp_path / "output.txt").exists()


def test_run_command_minimal_environment_does_not_inherit_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-sentinel")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-aws-sentinel")

    result = dispatch_command(
        Workspace(tmp_path),
        {
            "argv": [
                "python",
                "-c",
                "import os; print(os.getenv('DEEPSEEK_API_KEY')); "
                "print(os.getenv('AWS_SECRET_ACCESS_KEY'))",
            ]
        },
        lambda _: True,
    )

    payload = json.loads(result.content)
    assert result.is_error is False
    assert payload["stdout"] == f"None{os.linesep}None{os.linesep}"
    assert "fake-deepseek-sentinel" not in result.content
    assert "fake-aws-sentinel" not in result.content


def test_run_command_is_not_registered_in_existing_workspace_registry(tmp_path: Path) -> None:
    from coding_agent.file_tools import create_workspace_registry

    names = [spec.name for spec in create_workspace_registry(Workspace(tmp_path), allow_write=True)]

    assert names == ["list_files", "read_file", "write_file", "edit_file"]


def test_workspace_registry_only_exposes_run_command_when_enabled(tmp_path: Path) -> None:
    from coding_agent.file_tools import create_workspace_registry

    workspace = Workspace(tmp_path)
    default_names = [spec.name for spec in create_workspace_registry(workspace)]
    executable_names = [spec.name for spec in create_workspace_registry(workspace, allow_exec=True)]
    all_names = [
        spec.name
        for spec in create_workspace_registry(workspace, allow_write=True, allow_exec=True)
    ]

    assert default_names == ["list_files", "read_file"]
    assert executable_names == ["list_files", "read_file", "run_command"]
    assert all_names == ["list_files", "read_file", "write_file", "edit_file", "run_command"]
