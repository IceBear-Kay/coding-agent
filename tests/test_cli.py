from pathlib import Path

from coding_agent.cli import main
from coding_agent.models import ModelResponse, ToolCall
from coding_agent.provider import FakeProvider


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
    assert output == ["Project contents"]
    assert errors == []
    assert provider.requests[0][0][0].content == "Read README.md"


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
    assert output == ["Done"]


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
