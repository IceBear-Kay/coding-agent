import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from coding_agent.cli import build_parser, main
from coding_agent.models import ModelResponse, ToolCall
from coding_agent.provider import FakeProvider


def test_cli_parser_accepts_task_workspace_limits_and_tool_flags() -> None:
    args = build_parser().parse_args(
        [
            "检查项目",
            "--workspace",
            "D:/coding-agent-demo",
            "--allow-write",
            "--allow-exec",
            "--command-timeout",
            "3.5",
            "--command-output-limit",
            "2048",
            "--max-steps",
            "12",
            "--max-retries",
            "0",
        ]
    )

    assert args.task == "检查项目"
    assert args.workspace == "D:/coding-agent-demo"
    assert args.allow_write is True
    assert args.allow_exec is True
    assert args.command_timeout == 3.5
    assert args.command_output_limit == 2048
    assert args.max_steps == 12
    assert args.max_retries == 0


def test_cli_runs_injected_provider_and_passes_workspace_and_task(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Project contents", encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_readme",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="Project contents", finish_reason="stop"),
        ]
    )
    output: list[str] = []
    errors: list[str] = []

    exit_code = main(
        ["Read README.md", "--workspace", str(tmp_path), "--max-steps", "3"],
        provider=provider,
        output_fn=output.append,
        error_fn=errors.append,
    )

    assert exit_code == 0
    assert output[0].startswith("工具调用: read_file (call_readme)")
    assert output[1] == "工具结果 read_file: 已返回"
    assert output[2:] == ["Project contents", "停止原因: completed"]
    assert errors == []
    assert provider.requests[0][0][0].role == "system"
    assert provider.requests[0][0][1].content == "Read README.md"


def test_cli_prompts_for_task_when_argument_is_omitted(tmp_path: Path) -> None:
    provider = FakeProvider([ModelResponse(text="Done", finish_reason="stop")])
    prompts: list[str] = []
    output: list[str] = []

    exit_code = main(
        ["--workspace", str(tmp_path)],
        provider=provider,
        input_fn=lambda prompt: prompts.append(prompt) or "Inspect the workspace",
        output_fn=output.append,
    )

    assert exit_code == 0
    assert prompts == ["任务: "]
    assert output == ["Done", "停止原因: completed"]


def test_cli_reports_max_steps_without_requesting_another_response(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_list",
                        name="list_files",
                        arguments={},
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    errors: list[str] = []

    exit_code = main(
        ["Inspect files", "--workspace", str(tmp_path), "--max-steps", "1"],
        provider=provider,
        error_fn=errors.append,
    )

    assert exit_code == 1
    assert errors == ["停止原因: max_steps"]
    assert len(provider.requests) == 1


def test_cli_rejects_empty_interactive_task(tmp_path: Path) -> None:
    errors: list[str] = []

    exit_code = main(
        ["--workspace", str(tmp_path)],
        provider=FakeProvider([]),
        input_fn=lambda _: "  ",
        error_fn=errors.append,
    )

    assert exit_code == 2
    assert errors == ["错误: task 不能为空"]


def test_cli_handles_keyboard_interrupt_during_input(tmp_path: Path) -> None:
    errors: list[str] = []

    def interrupt(_: str) -> str:
        raise KeyboardInterrupt

    exit_code = main(
        ["--workspace", str(tmp_path)],
        provider=FakeProvider([]),
        input_fn=interrupt,
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]


def test_cli_handles_keyboard_interrupt_during_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []

    def interrupt_workspace(_: str | Path) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("coding_agent.cli.Workspace", interrupt_workspace)

    exit_code = main(
        ["Inspect files", "--workspace", str(tmp_path)],
        provider=FakeProvider([]),
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]


def test_cli_handles_keyboard_interrupt_during_output(tmp_path: Path) -> None:
    provider = FakeProvider([ModelResponse(text="Done", finish_reason="stop")])
    errors: list[str] = []

    def interrupt_output(_: str) -> None:
        raise KeyboardInterrupt

    exit_code = main(
        ["Finish the task", "--workspace", str(tmp_path)],
        provider=provider,
        output_fn=interrupt_output,
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]


def test_cli_defaults_to_read_only_tools_even_if_model_requests_write(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "created.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="Write was unavailable.", finish_reason="stop"),
        ]
    )
    approval_prompts: list[str] = []

    exit_code = main(
        ["Create a file", "--workspace", str(tmp_path), "--max-steps", "3"],
        provider=provider,
        input_fn=lambda prompt: approval_prompts.append(prompt) or "y",
    )

    assert exit_code == 0
    assert approval_prompts == []
    assert not (tmp_path / "created.txt").exists()
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == [
        "list_files",
        "read_file",
    ]
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert tool_message.tool_call_id == "call_write"
    assert "Unknown tool" in tool_message.content


def test_cli_executes_approved_command_with_configured_limits(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_run",
                        name="run_command",
                        arguments={
                            "argv": ["python", "-c", "print('command output')"],
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="命令已完成。", finish_reason="stop"),
        ]
    )
    output: list[str] = []

    exit_code = main(
        [
            "Run a short Python command",
            "--workspace",
            str(tmp_path),
            "--allow-exec",
            "--command-timeout",
            "2",
            "--command-output-limit",
            "1024",
            "--max-steps",
            "3",
        ],
        provider=provider,
        input_fn=lambda _: "y",
        output_fn=output.append,
    )

    assert exit_code == 0
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == [
        "list_files",
        "read_file",
        "run_command",
    ]
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    payload = json.loads(tool_message.content)
    assert payload["status"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["stdout"].strip() == "command output"
    assert any("Timeout seconds: 2.0" in message for message in output)
    assert any("Combined output limit bytes: 1024" in message for message in output)
    assert output[-2:] == ["命令已完成。", "停止原因: completed"]


def test_cli_does_not_expose_run_command_without_allow_exec(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_run",
                        name="run_command",
                        arguments={"argv": ["python", "-c", "print('not run')"]},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="命令不可用。", finish_reason="stop"),
        ]
    )

    exit_code = main(
        ["Run a command", "--workspace", str(tmp_path)],
        provider=provider,
        input_fn=lambda _: "y",
    )

    assert exit_code == 0
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == [
        "list_files",
        "read_file",
    ]
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert "Unknown tool" in tool_message.content


def test_cli_rejects_invalid_command_limits_before_provider_call(tmp_path: Path) -> None:
    provider = FakeProvider([])
    errors: list[str] = []

    exit_code = main(
        [
            "Run a command",
            "--workspace",
            str(tmp_path),
            "--allow-exec",
            "--command-timeout",
            "61",
        ],
        provider=provider,
        error_fn=errors.append,
    )

    assert exit_code == 2
    assert provider.requests == []
    assert "timeout_seconds" in errors[0]


def test_cli_default_input_rejects_redirected_command_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_run",
                        name="run_command",
                        arguments={
                            "argv": [
                                "python",
                                "-c",
                                "open('not-created.txt', 'w').write('unexpected')",
                            ]
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="命令已拒绝。", finish_reason="stop"),
        ]
    )
    redirected_input = StringIO("y\n")
    output: list[str] = []
    monkeypatch.setattr(sys, "stdin", redirected_input)

    exit_code = main(
        ["Run a command", "--workspace", str(tmp_path), "--allow-exec"],
        provider=provider,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert redirected_input.tell() == 0
    assert not (tmp_path / "not-created.txt").exists()
    assert "审批结果: 已拒绝（非交互输入不能用于审批）" in output
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert json.loads(tool_message.content)["status"] == "denied"


def test_cli_approves_write_and_edit_as_separate_operations(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "program.py", "content": "value = 1\n"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_edit",
                        name="edit_file",
                        arguments={
                            "path": "program.py",
                            "old_text": "value = 1",
                            "new_text": "value = 2",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="文件已创建并修改。", finish_reason="stop"),
        ]
    )
    answers = iter(["y", "yes"])
    prompts: list[str] = []
    output: list[str] = []

    exit_code = main(
        [
            "Create and update program.py",
            "--workspace",
            str(tmp_path),
            "--allow-write",
            "--max-steps",
            "4",
        ],
        provider=provider,
        input_fn=lambda prompt: prompts.append(prompt) or next(answers),
        output_fn=output.append,
    )

    assert exit_code == 0
    assert prompts == ["批准本次操作？[y/N]: ", "批准本次操作？[y/N]: "]
    assert (tmp_path / "program.py").read_text(encoding="utf-8") == "value = 2\n"
    assert [schema["function"]["name"] for schema in provider.requests[0][1]] == [
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
    ]
    first_tool_message = next(
        message for message in provider.requests[1][0] if message.tool_call_id == "call_write"
    )
    second_tool_message = next(
        message for message in provider.requests[2][0] if message.tool_call_id == "call_edit"
    )
    assert json.loads(first_tool_message.content)["status"] == "created"
    assert json.loads(second_tool_message.content)["status"] == "edited"
    assert any("Path: program.py" in message for message in output)
    assert output[-2:] == ["文件已创建并修改。", "停止原因: completed"]


def test_cli_default_input_rejects_redirected_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "redirected.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="操作未执行。", finish_reason="stop"),
        ]
    )
    redirected_input = StringIO("y\n")
    output: list[str] = []
    monkeypatch.setattr(sys, "stdin", redirected_input)

    exit_code = main(
        ["Create redirected.txt", "--workspace", str(tmp_path), "--allow-write"],
        provider=provider,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert redirected_input.tell() == 0
    assert not (tmp_path / "redirected.txt").exists()
    assert "审批结果: 已拒绝（非交互输入不能用于审批）" in output
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert json.loads(tool_message.content)["status"] == "denied"


@pytest.mark.parametrize("answer", ["", "n"])
def test_cli_rejected_or_empty_approval_does_not_write_file(
    tmp_path: Path,
    answer: str,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "denied.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="操作未执行。", finish_reason="stop"),
        ]
    )

    exit_code = main(
        ["Create denied.txt", "--workspace", str(tmp_path), "--allow-write"],
        provider=provider,
        input_fn=lambda _: answer,
    )

    assert exit_code == 0
    assert not (tmp_path / "denied.txt").exists()
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert json.loads(tool_message.content)["status"] == "denied"


def test_cli_eof_during_approval_denies_operation(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "denied.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="操作未执行。", finish_reason="stop"),
        ]
    )
    output: list[str] = []

    def end_input(_: str) -> str:
        raise EOFError

    exit_code = main(
        ["Create denied.txt", "--workspace", str(tmp_path), "--allow-write"],
        provider=provider,
        input_fn=end_input,
        output_fn=output.append,
    )

    assert exit_code == 0
    assert not (tmp_path / "denied.txt").exists()
    assert "审批结果: 已拒绝（无法读取输入）" in output
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert json.loads(tool_message.content)["status"] == "denied"


def test_cli_keyboard_interrupt_during_approval_stops_without_writing(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "interrupted.txt", "content": "content"},
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    errors: list[str] = []

    def interrupt(_: str) -> str:
        raise KeyboardInterrupt

    exit_code = main(
        ["Create interrupted.txt", "--workspace", str(tmp_path), "--allow-write"],
        provider=provider,
        input_fn=interrupt,
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]
    assert not (tmp_path / "interrupted.txt").exists()
    assert len(provider.requests) == 1


def test_cli_keyboard_interrupt_during_edit_cleans_up_and_returns_130(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_edit",
                        name="edit_file",
                        arguments={
                            "path": "notes.txt",
                            "old_text": "before",
                            "new_text": "after",
                        },
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    errors: list[str] = []

    def interrupt_fsync(_: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("coding_agent.file_tools.os.fsync", interrupt_fsync)

    exit_code = main(
        ["Edit notes.txt", "--workspace", str(tmp_path), "--allow-write"],
        provider=provider,
        input_fn=lambda _: "y",
        error_fn=errors.append,
    )

    assert exit_code == 130
    assert errors == ["停止原因: interrupted"]
    assert target.read_text(encoding="utf-8") == "before"
    assert list(tmp_path.glob(".notes.txt.*.tmp")) == []
    assert len(provider.requests) == 1
