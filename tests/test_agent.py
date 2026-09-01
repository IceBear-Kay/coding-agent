import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from coding_agent import (
    COMPLETED_STOP_REASON,
    CONTEXT_LIMIT_STOP_REASON,
    DEFAULT_SYSTEM_PROMPT,
    AgentEvent,
    AgentLoop,
    AgentSession,
    AgentState,
    ApprovalRequest,
    CommandLimits,
    FakeProvider,
    Message,
    ModelResponse,
    ToolCall,
    Workspace,
    create_read_only_registry,
    create_workspace_registry,
    measure_context_bytes,
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


def test_agent_loop_records_tool_result_before_result_event_interrupt(tmp_path: Path) -> None:
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
                        id="call_interrupted_write",
                        name="write_file",
                        arguments={"path": "created.txt", "content": "saved\n"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="这一步不应被请求。", finish_reason="stop"),
        ]
    )
    events: list[AgentEvent] = []

    def interrupt_after_result(event: AgentEvent) -> None:
        events.append(event)
        if event.kind == "tool_result":
            raise KeyboardInterrupt

    result = AgentLoop(
        provider,
        workspace,
        registry=registry,
        event_callback=interrupt_after_result,
    ).run("创建 created.txt。")

    assert result.stop_reason == "interrupted"
    assert isinstance(result.error, KeyboardInterrupt)
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "saved\n"
    assert result.state.messages[-1] == Message(
        role="tool",
        tool_call_id="call_interrupted_write",
        content='{"bytes_written":6,"path":"created.txt","status":"created"}',
    )
    assert [event.kind for event in events] == ["tool_call", "tool_result"]
    assert len(provider.requests) == 1


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


def test_openai_provider_round_trips_side_effect_tools_and_actual_outputs(
    tmp_path: Path,
) -> None:
    request_bodies: list[dict[str, Any]] = []
    tool_steps = [
        (
            "call_http_write",
            "write_file",
            {"path": "answer.py", "content": "print(0)\n"},
            "创建初始程序。",
        ),
        (
            "call_http_run_wrong",
            "run_command",
            {"argv": ["python", "answer.py"], "cwd": ".", "stdin": ""},
            "执行样例并读取实际输出。",
        ),
        (
            "call_http_edit",
            "edit_file",
            {"path": "answer.py", "old_text": "print(0)", "new_text": "print(7)"},
            "退出码为零但答案不是预期值，进行精确修正。",
        ),
        (
            "call_http_run_fixed",
            "run_command",
            {"argv": ["python", "answer.py"], "cwd": ".", "stdin": ""},
            "再次运行并确认实际输出。",
        ),
    ]

    def response_for_tool(
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        reasoning_content: str,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": reasoning_content,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(
                                            arguments,
                                            ensure_ascii=False,
                                            separators=(",", ":"),
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        response_index = len(request_bodies) - 1
        if response_index < len(tool_steps):
            call_id, tool_name, arguments, reasoning_content = tool_steps[response_index]
            return response_for_tool(call_id, tool_name, arguments, reasoning_content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "已根据实际输出修正并确认答案为 7。",
                            "reasoning_content": "实际输出已经等于预期值。",
                        },
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
    workspace = Workspace(tmp_path)
    approvals: list[ApprovalRequest] = []
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        allow_exec=True,
        approval_callback=lambda request: approvals.append(request) or True,
        command_limits=CommandLimits(timeout_seconds=2),
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(config, client)
        result = AgentLoop(
            provider,
            workspace,
            registry=registry,
            system_prompt="通用测试策略",
            max_steps=8,
        ).run("创建程序，比较实际输出并修正到预期答案 7。")

    assert result.answer == "已根据实际输出修正并确认答案为 7。"
    assert result.stop_reason == "completed"
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "print(7)\n"
    assert [request.operation for request in approvals] == [
        "write_file",
        "run_command",
        "edit_file",
        "run_command",
    ]
    assert len(request_bodies) == 5

    schema_names = {tool["function"]["name"] for tool in request_bodies[0]["tools"]}
    assert schema_names == {
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
    }

    for body, (_, _, _, expected_reasoning) in zip(request_bodies[1:5], tool_steps, strict=True):
        assistant_messages = [
            message for message in body["messages"] if message["role"] == "assistant"
        ]
        assert assistant_messages[-1]["reasoning_content"] == expected_reasoning

    expected_arguments = [arguments for _, _, arguments, _ in tool_steps]
    for index, (call_id, _, arguments, _) in enumerate(tool_steps, start=1):
        assistant_message = next(
            message
            for message in request_bodies[index]["messages"]
            if message.get("role") == "assistant"
            and message.get("tool_calls")
            and message["tool_calls"][0]["id"] == call_id
        )
        assert assistant_message["tool_calls"][0]["function"]["arguments"] == json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        tool_message = next(
            message
            for message in request_bodies[index]["messages"]
            if message.get("tool_call_id") == call_id
        )
        payload = json.loads(tool_message["content"])
        assert tool_message["role"] == "tool"
        assert payload["status"] in {"created", "completed", "edited"}

    first_run_payload = json.loads(
        next(
            message
            for message in request_bodies[2]["messages"]
            if message.get("tool_call_id") == "call_http_run_wrong"
        )["content"]
    )
    second_run_payload = json.loads(
        next(
            message
            for message in request_bodies[4]["messages"]
            if message.get("tool_call_id") == "call_http_run_fixed"
        )["content"]
    )
    assert first_run_payload["status"] == "completed"
    assert first_run_payload["exit_code"] == 0
    assert first_run_payload["stdout"].strip() != "7"
    assert second_run_payload["status"] == "completed"
    assert second_run_payload["exit_code"] == 0
    assert second_run_payload["stdout"].strip() == "7"
    assert expected_arguments[0]["content"] == "print(0)\n"


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


def test_agent_loop_stops_before_first_provider_call_when_context_exceeds_budget(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([ModelResponse(text="should not run", finish_reason="stop")])
    loop = AgentLoop(
        provider,
        Workspace(tmp_path),
        max_context_bytes=1,
    )

    result = loop.run("This request is larger than one byte.")

    assert result.stop_reason == CONTEXT_LIMIT_STOP_REASON
    assert result.state.step_count == 0
    assert provider.requests == []
    assert result.error is not None
    assert "This request" not in str(result.error)
    assert "context input exceeds byte budget" in str(result.error)


def test_agent_loop_keeps_tool_result_when_next_context_check_exceeds_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("tool result", encoding="utf-8")
    workspace = Workspace(tmp_path)
    registry = create_read_only_registry(workspace)
    task = "Read notes.txt"
    initial_size = measure_context_bytes([Message(role="user", content=task)], registry.schemas())
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_notes",
                        name="read_file",
                        arguments={"path": "notes.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="should not be requested", finish_reason="stop"),
        ]
    )
    loop = AgentLoop(
        provider,
        workspace,
        registry=registry,
        max_context_bytes=initial_size,
    )

    result = loop.run(task)

    assert result.stop_reason == CONTEXT_LIMIT_STOP_REASON
    assert result.state.step_count == 1
    assert len(provider.requests) == 1
    tool_message = next(message for message in result.state.messages if message.role == "tool")
    assert tool_message.tool_call_id == "call_notes"
    assert tool_message.content == "tool result"


def test_agent_loop_keeps_completed_write_when_context_budget_is_exceeded(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        approval_callback=lambda _: True,
    )
    task = "创建 once.py。"
    tool_call = ToolCall(
        id="call_write_once",
        name="write_file",
        arguments={"path": "once.py", "content": "print(1)\n"},
    )
    expected_tool_content = '{"bytes_written":9,"path":"once.py","status":"created"}'
    messages_after_tool = [
        Message(role="user", content=task),
        Message(role="assistant", content=None, tool_calls=[tool_call]),
        Message(role="tool", tool_call_id=tool_call.id, content=expected_tool_content),
    ]
    max_context_bytes = measure_context_bytes(messages_after_tool, registry.schemas()) - 1
    provider = FakeProvider(
        [
            ModelResponse(tool_calls=[tool_call], finish_reason="tool_calls"),
            ModelResponse(text="不应再次请求", finish_reason="stop"),
        ]
    )

    result = AgentLoop(
        provider,
        workspace,
        registry=registry,
        max_context_bytes=max_context_bytes,
    ).run(task)

    assert result.stop_reason == CONTEXT_LIMIT_STOP_REASON
    assert result.state.step_count == 1
    assert len(provider.requests) == 1
    assert (tmp_path / "once.py").read_text(encoding="utf-8") == "print(1)\n"
    assert [
        message.tool_call_id for message in result.state.messages if message.role == "tool"
    ] == ["call_write_once"]


def test_agent_loop_checks_context_budget_before_each_provider_retry(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [ProviderNetworkError("temporary"), ModelResponse(text="完成", finish_reason="stop")]
    )
    loop = AgentLoop(
        provider,
        Workspace(tmp_path),
        max_steps=2,
        max_retries=1,
    )
    checks: list[int] = []
    original_budget = loop.context_budget

    class RecordingBudget:
        max_bytes = original_budget.max_bytes

        def check(self, messages, tools):
            result = original_budget.check(messages, tools)
            checks.append(result.used_bytes)
            return result

    loop.context_budget = RecordingBudget()

    result = loop.run("重试一次。")

    assert result.stop_reason == COMPLETED_STOP_REASON
    assert result.state.step_count == 2
    assert len(provider.requests) == 2
    assert len(checks) == 2
    assert checks[0] == checks[1]


def test_agent_loop_trim_policy_uses_recent_context_without_mutating_session_history(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="旧任务已完成", finish_reason="stop"),
            ModelResponse(text="新任务已完成", finish_reason="stop"),
        ]
    )
    workspace = Workspace(tmp_path)
    registry = create_read_only_registry(workspace)
    system = Message(role="system", content=DEFAULT_SYSTEM_PROMPT)
    current = Message(role="user", content="新任务")
    budget = measure_context_bytes([system, current], registry.schemas())
    session = AgentSession(
        AgentLoop(
            provider,
            workspace,
            registry=registry,
            max_context_bytes=budget,
            context_policy="trim",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
    )

    first = session.run("旧任务")
    second = session.run("新任务")

    assert first.stop_reason == COMPLETED_STOP_REASON
    assert second.stop_reason == COMPLETED_STOP_REASON
    assert len(provider.requests) == 2
    assert [message.content for message in provider.requests[1][0]] == [
        DEFAULT_SYSTEM_PROMPT,
        "新任务",
    ]
    assert [message.content for message in session.messages if message.role == "user"] == [
        "旧任务",
        "新任务",
    ]


def test_agent_loop_trim_policy_removes_old_tool_exchange_as_one_group(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_old",
                        name="read_file",
                        arguments={"path": "old.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="旧任务已完成", finish_reason="stop"),
            ModelResponse(text="新任务已完成", finish_reason="stop"),
        ]
    )
    (tmp_path / "old.txt").write_text("old", encoding="utf-8")
    workspace = Workspace(tmp_path)
    registry = create_read_only_registry(workspace)
    budget = measure_context_bytes(
        [
            Message(role="system", content=DEFAULT_SYSTEM_PROMPT),
            Message(role="user", content="旧任务"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_old",
                        name="read_file",
                        arguments={"path": "old.txt"},
                    )
                ],
            ),
            Message(role="tool", content="old", tool_call_id="call_old"),
            Message(role="assistant", content="旧任务已完成"),
        ],
        registry.schemas(),
    )
    session = AgentSession(
        AgentLoop(
            provider,
            workspace,
            registry=registry,
            max_context_bytes=budget,
            context_policy="trim",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
    )

    assert session.run("旧任务").stop_reason == COMPLETED_STOP_REASON
    assert session.run("新任务").stop_reason == COMPLETED_STOP_REASON

    assert len(provider.requests) == 3
    second_request = provider.requests[1][0]
    assert any(message.role == "tool" for message in second_request)
    third_request = provider.requests[2][0]
    assert [message.role for message in third_request] == ["system", "user"]
    assert all(message.tool_call_id != "call_old" for message in third_request)


def test_agent_loop_trim_policy_does_not_remove_current_task_after_tool_result_exceeds_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "notes.txt").write_text("tool result", encoding="utf-8")
    workspace = Workspace(tmp_path)
    registry = create_read_only_registry(workspace)
    task = "读取 notes.txt"
    after_tool = [
        Message(role="user", content=task),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="call_notes", name="read_file", arguments={"path": "notes.txt"})
            ],
        ),
        Message(role="tool", content="tool result", tool_call_id="call_notes"),
    ]
    max_context_bytes = measure_context_bytes(after_tool, registry.schemas()) - 1
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_notes",
                        name="read_file",
                        arguments={"path": "notes.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="不应请求", finish_reason="stop"),
        ]
    )

    result = AgentLoop(
        provider,
        workspace,
        registry=registry,
        max_context_bytes=max_context_bytes,
        context_policy="trim",
    ).run(task)

    assert result.stop_reason == CONTEXT_LIMIT_STOP_REASON
    assert result.state.step_count == 1
    assert len(provider.requests) == 1
    assert result.state.messages[-1].tool_call_id == "call_notes"
