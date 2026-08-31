import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from coding_agent import (
    AgentEvent,
    AgentLoop,
    AgentState,
    ApprovalRequest,
    CommandLimits,
    FakeProvider,
    Message,
    ModelResponse,
    ToolCall,
    Workspace,
    create_workspace_registry,
)
from coding_agent.config import ProviderConfig
from coding_agent.errors import (
    ProviderAuthenticationError,
    ProviderNetworkError,
)
from coding_agent.provider import OpenAICompatibleProvider


class ScriptedProvider:
    """Provider double that can return responses or raise scripted failures."""

    def __init__(self, outcomes: list[ModelResponse | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[tuple[list[Message], list[dict[str, Any]]]] = []

    def complete(
        self,
        messages: list[Message],
        tool_schemas: list[dict[str, Any]],
    ) -> ModelResponse:
        self.requests.append(
            ([message.model_copy(deep=True) for message in messages], deepcopy(tool_schemas))
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome.model_copy(deep=True)


def test_agent_loop_runs_tool_then_returns_final_answer(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Project contents", encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                text=None,
                reasoning_content="I should inspect the README.",
                tool_calls=[
                    ToolCall(
                        id="call_readme",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                text="The README says: Project contents",
                reasoning_content="The file was read successfully.",
                finish_reason="stop",
            ),
        ]
    )
    loop = AgentLoop(provider, Workspace(tmp_path), max_steps=3)

    result = loop.run("Read README.md and summarize it.")

    assert result.answer == "The README says: Project contents"
    assert result.final_answer == result.answer
    assert result.stop_reason == "completed"
    assert result.state.step_count == 2
    assert result.state.messages == [
        Message(role="user", content="Read README.md and summarize it."),
        Message(
            role="assistant",
            content=None,
            reasoning_content="I should inspect the README.",
            tool_calls=[
                ToolCall(
                    id="call_readme",
                    name="read_file",
                    arguments={"path": "README.md"},
                )
            ],
        ),
        Message(
            role="tool",
            tool_call_id="call_readme",
            content="Project contents",
        ),
        Message(
            role="assistant",
            content="The README says: Project contents",
            reasoning_content="The file was read successfully.",
        ),
    ]
    assert provider.requests[1][0] == result.state.messages[:3]
    assert provider.requests[0][0] == result.state.messages[:1]


def test_agent_loop_emits_real_tool_events_in_dispatch_order(tmp_path: Path) -> None:
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
            ModelResponse(text="Done", finish_reason="stop"),
        ]
    )
    events: list[AgentEvent] = []

    result = AgentLoop(
        provider,
        Workspace(tmp_path),
        event_callback=events.append,
    ).run("Read README.md.")

    assert result.stop_reason == "completed"
    assert [event.kind for event in events] == ["tool_call", "tool_result"]
    assert events[0].tool_call == ToolCall(
        id="call_readme",
        name="read_file",
        arguments={"path": "README.md"},
    )
    assert events[1].tool_result is not None
    assert events[1].tool_name == "read_file"
    assert events[1].tool_result.tool_call_id == "call_readme"
    assert events[1].tool_result.content == "Project contents"


def test_agent_loop_round_trips_approved_file_result(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        approval_callback=lambda _: True,
    )
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        name="write_file",
                        arguments={"path": "created.txt", "content": "created"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="Created the file.", finish_reason="stop"),
        ]
    )

    result = AgentLoop(provider, workspace, registry=registry, max_steps=3).run(
        "Create created.txt"
    )

    assert result.stop_reason == "completed"
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created"
    tool_message = next(message for message in provider.requests[1][0] if message.role == "tool")
    assert tool_message.tool_call_id == "call_write"
    assert json.loads(tool_message.content)["status"] == "created"


def test_agent_loop_completes_real_coding_repair_cycle(tmp_path: Path) -> None:
    (tmp_path / "problem.txt").write_text(
        "Read two integers and print their sum.",
        encoding="utf-8",
    )
    workspace = Workspace(tmp_path)
    approvals: list[ApprovalRequest] = []

    def approve(request: ApprovalRequest) -> bool:
        approvals.append(request)
        return True

    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        allow_exec=True,
        approval_callback=approve,
        command_limits=CommandLimits(timeout_seconds=2),
    )
    provider = FakeProvider(
        [
            ModelResponse(
                reasoning_content="先读取题目。",
                tool_calls=[
                    ToolCall(
                        id="call_read_problem",
                        name="read_file",
                        arguments={"path": "problem.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write_solution",
                        name="write_file",
                        arguments={
                            "path": "solution.py",
                            "content": "import sys\nprint(unknown_name)\n",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_run_bad",
                        name="run_command",
                        arguments={
                            "argv": ["python", "solution.py"],
                            "stdin": "1 2\n",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_edit_solution",
                        name="edit_file",
                        arguments={
                            "path": "solution.py",
                            "old_text": "unknown_name",
                            "new_text": "sum(map(int, input().split()))",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_run_fixed",
                        name="run_command",
                        arguments={
                            "argv": ["python", "solution.py"],
                            "stdin": "1 2\n",
                        },
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="程序已修正并通过样例。", finish_reason="stop"),
        ]
    )

    result = AgentLoop(provider, workspace, registry=registry, max_steps=8).run(
        "读取题目，编写程序并运行样例；如果失败就修正后再次运行。"
    )

    assert result.answer == "程序已修正并通过样例。"
    assert result.stop_reason == "completed"
    assert (tmp_path / "solution.py").read_text(encoding="utf-8") == (
        "import sys\nprint(sum(map(int, input().split())))\n"
    )
    assert [request.operation for request in approvals] == [
        "write_file",
        "run_command",
        "edit_file",
        "run_command",
    ]
    failed_run = next(
        message for message in provider.requests[3][0] if message.tool_call_id == "call_run_bad"
    )
    failed_payload = json.loads(failed_run.content)
    assert failed_payload["status"] == "failed"
    assert "NameError" in failed_payload["stderr"]
    fixed_run = next(
        message for message in provider.requests[5][0] if message.tool_call_id == "call_run_fixed"
    )
    fixed_payload = json.loads(fixed_run.content)
    assert fixed_payload["status"] == "completed"
    assert fixed_payload["stdout"].strip() == "3"
    assert provider.requests[3][0][-1].tool_call_id == "call_run_bad"
    assert provider.requests[5][0][-1].tool_call_id == "call_run_fixed"


def test_agent_loop_rejection_has_no_side_effect_and_returns_to_model(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    approvals: list[ApprovalRequest] = []
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        approval_callback=lambda request: approvals.append(request) or False,
    )
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_denied_write",
                        name="write_file",
                        arguments={"path": "denied.py", "content": "print(1)\n"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="用户拒绝了写入。", finish_reason="stop"),
        ]
    )

    result = AgentLoop(provider, workspace, registry=registry).run("创建 denied.py。")

    assert result.answer == "用户拒绝了写入。"
    assert not (tmp_path / "denied.py").exists()
    assert [request.operation for request in approvals] == ["write_file"]
    denied_message = provider.requests[1][0][-1]
    assert json.loads(denied_message.content)["status"] == "denied"
    assert denied_message.tool_call_id == "call_denied_write"


def test_agent_loop_transient_retry_does_not_repeat_completed_write(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        approval_callback=lambda _: True,
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_write_once",
                        name="write_file",
                        arguments={"path": "once.py", "content": "print(1)\n"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ProviderNetworkError("temporary after write"),
            ModelResponse(text="写入已完成。", finish_reason="stop"),
        ]
    )

    result = AgentLoop(
        provider,
        workspace,
        registry=registry,
        max_steps=4,
        max_retries=1,
    ).run("创建 once.py。")

    assert result.stop_reason == "completed"
    assert (tmp_path / "once.py").read_text(encoding="utf-8") == "print(1)\n"
    assert len(provider.requests) == 3
    assert provider.requests[1][0] == provider.requests[2][0]


def test_agent_loop_returns_real_command_timeout_to_provider(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    registry = create_workspace_registry(
        workspace,
        allow_exec=True,
        approval_callback=lambda _: True,
        command_limits=CommandLimits(timeout_seconds=0.1),
    )
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_timeout",
                        name="run_command",
                        arguments={"argv": ["python", "-c", "import time; time.sleep(3)"]},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="命令超时，任务未完成。", finish_reason="stop"),
        ]
    )

    result = AgentLoop(provider, workspace, registry=registry, max_steps=3).run("运行一个短命令。")

    assert result.stop_reason == "completed"
    timeout_message = provider.requests[1][0][-1]
    assert json.loads(timeout_message.content)["status"] == "timeout"
    assert timeout_message.tool_call_id == "call_timeout"


def test_agent_loop_with_openai_provider_round_trips_tool_protocol(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("协议测试", encoding="utf-8")
    request_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        request_bodies.append(body)
        if len(request_bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "读取 README。",
                                "tool_calls": [
                                    {
                                        "id": "call_http_read",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"README.md"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "协议闭环完成。"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    config = ProviderConfig(
        api_key="test-key",
        base_url="https://api.example.com/v1",
        model="test-model",
        timeout_seconds=5,
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(config, client)
        result = AgentLoop(
            provider,
            Workspace(tmp_path),
            system_prompt="通用测试策略",
        ).run("读取 README.md。")

    assert result.answer == "协议闭环完成。"
    assert result.stop_reason == "completed"
    assert request_bodies[0]["messages"][0] == {"role": "system", "content": "通用测试策略"}
    assert request_bodies[1]["messages"][1]["role"] == "user"
    assistant_message = request_bodies[1]["messages"][2]
    assert assistant_message["reasoning_content"] == "读取 README。"
    assert assistant_message["tool_calls"][0]["id"] == "call_http_read"
    assert assistant_message["tool_calls"][0]["function"]["arguments"] == '{"path":"README.md"}'
    tool_message = request_bodies[1]["messages"][3]
    assert tool_message == {
        "role": "tool",
        "content": "协议测试",
        "tool_call_id": "call_http_read",
    }


def test_agent_loop_stops_at_max_steps_after_executing_tool(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Project contents", encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="call_readme",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    loop = AgentLoop(provider, Workspace(tmp_path), max_steps=1)

    result = loop.run("Inspect the README.")

    assert result.answer is None
    assert result.stop_reason == "max_steps"
    assert result.state.step_count == 1
    assert [message.role for message in result.state.messages] == ["user", "assistant", "tool"]
    assert result.state.messages[-1].tool_call_id == "call_readme"
    assert len(provider.requests) == 1


def test_agent_loop_rejects_invalid_task_and_step_limit(tmp_path: Path) -> None:
    provider = FakeProvider([ModelResponse(text="Done")])
    workspace = Workspace(tmp_path)

    with pytest.raises(ValueError, match="max_steps"):
        AgentLoop(provider, workspace, max_steps=0)

    loop = AgentLoop(provider, workspace)
    with pytest.raises(ValueError, match="task"):
        loop.run(" ")


def test_agent_state_is_available_after_run(tmp_path: Path) -> None:
    loop = AgentLoop(FakeProvider([ModelResponse(text="Done")]), Workspace(tmp_path))

    result = loop.run("Finish immediately.")

    assert loop.state is result.state
    assert isinstance(loop.state, AgentState)


def test_agent_loop_returns_multiple_tool_results_in_call_order(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Project contents", encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                text=None,
                reasoning_content="I need two observations.",
                tool_calls=[
                    ToolCall(
                        id="call_missing",
                        name="missing_tool",
                        arguments={},
                    ),
                    ToolCall(
                        id="call_readme",
                        name="read_file",
                        arguments={"path": "README.md"},
                    ),
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="I received both tool results.", finish_reason="stop"),
        ]
    )
    result = AgentLoop(provider, Workspace(tmp_path), max_steps=3).run("Inspect the workspace.")

    assert result.answer == "I received both tool results."
    assert result.state.messages[1].reasoning_content == "I need two observations."
    tool_messages = [message for message in result.state.messages if message.role == "tool"]
    assert [message.tool_call_id for message in tool_messages] == ["call_missing", "call_readme"]
    assert "Unknown tool" in tool_messages[0].content
    assert tool_messages[1].content == "Project contents"
    assert [
        message.tool_call_id for message in provider.requests[1][0] if message.role == "tool"
    ] == [
        "call_missing",
        "call_readme",
    ]


def test_agent_loop_retries_transient_provider_error_within_budget(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [ProviderNetworkError("temporary"), ModelResponse(text="Done", finish_reason="stop")]
    )
    delays: list[float] = []
    loop = AgentLoop(
        provider,
        Workspace(tmp_path),
        max_steps=3,
        max_retries=2,
        retry_delay_seconds=0.5,
        sleep=delays.append,
    )

    result = loop.run("Finish the task.")

    assert result.answer == "Done"
    assert result.stop_reason == "completed"
    assert result.error is None
    assert result.state.step_count == 2
    assert len(provider.requests) == 2
    assert delays == [0.5]


def test_agent_loop_does_not_retry_past_max_steps(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            ProviderNetworkError("temporary 1"),
            ProviderNetworkError("temporary 2"),
            ModelResponse(text="Should not run"),
        ]
    )
    loop = AgentLoop(
        provider,
        Workspace(tmp_path),
        max_steps=2,
        max_retries=10,
    )

    result = loop.run("Finish the task.")

    assert result.answer is None
    assert result.stop_reason == "transient_provider_error"
    assert isinstance(result.error, ProviderNetworkError)
    assert result.state.step_count == 2
    assert len(provider.requests) == 2


def test_agent_loop_stops_immediately_on_fatal_provider_error(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [ProviderAuthenticationError("invalid credentials"), ModelResponse(text="Should not run")]
    )
    loop = AgentLoop(provider, Workspace(tmp_path), max_steps=3, max_retries=5)

    result = loop.run("Finish the task.")

    assert result.answer is None
    assert result.stop_reason == "fatal_error"
    assert isinstance(result.error, ProviderAuthenticationError)
    assert result.state.step_count == 1
    assert len(provider.requests) == 1


def test_agent_loop_converts_unexpected_provider_error_to_stop_result(tmp_path: Path) -> None:
    provider = ScriptedProvider([RuntimeError("unexpected")])

    result = AgentLoop(provider, Workspace(tmp_path)).run("Finish the task.")

    assert result.answer is None
    assert result.stop_reason == "provider_error"
    assert isinstance(result.error, RuntimeError)


def test_agent_loop_handles_keyboard_interrupt_without_traceback(tmp_path: Path) -> None:
    provider = ScriptedProvider([KeyboardInterrupt()])

    result = AgentLoop(provider, Workspace(tmp_path)).run("Stop me safely.")

    assert result.answer is None
    assert result.stop_reason == "interrupted"
    assert isinstance(result.error, KeyboardInterrupt)
    assert result.state.step_count == 1


@pytest.mark.parametrize(
    "finish_reason", ["length", "content_filter", "insufficient_system_resource"]
)
def test_agent_loop_preserves_non_normal_finish_reason(finish_reason: str, tmp_path: Path) -> None:
    provider = FakeProvider([ModelResponse(text=None, finish_reason=finish_reason)])

    result = AgentLoop(provider, Workspace(tmp_path)).run("Answer if possible.")

    assert result.answer is None
    assert result.stop_reason == finish_reason
    assert result.stop_reason != "completed"


@pytest.mark.parametrize(
    "finish_reason", ["length", "content_filter", "insufficient_system_resource"]
)
def test_agent_loop_does_not_dispatch_tools_for_non_normal_finish_reason(
    finish_reason: str,
    tmp_path: Path,
) -> None:
    class RecordingDispatcher:
        def __init__(self) -> None:
            self.calls: list[ToolCall] = []

        def dispatch(self, tool_call: ToolCall) -> None:
            self.calls.append(tool_call)

    dispatcher = RecordingDispatcher()
    provider = FakeProvider(
        [
            ModelResponse(
                text=None,
                reasoning_content="The response was cut off.",
                tool_calls=[
                    ToolCall(
                        id="call_should_not_run",
                        name="list_files",
                        arguments={},
                    )
                ],
                finish_reason=finish_reason,
            )
        ]
    )
    loop = AgentLoop(provider, Workspace(tmp_path), dispatcher=dispatcher, max_steps=3)

    result = loop.run("Inspect files.")

    assert result.stop_reason == finish_reason
    assert result.state.messages[-1].role == "assistant"
    assert result.state.messages[-1].tool_calls[0].id == "call_should_not_run"
    assert not dispatcher.calls
    assert len(provider.requests) == 1
