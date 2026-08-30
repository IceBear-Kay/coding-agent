from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from coding_agent import (
    AgentLoop,
    AgentState,
    FakeProvider,
    Message,
    ModelResponse,
    ToolCall,
    Workspace,
)
from coding_agent.errors import (
    ProviderAuthenticationError,
    ProviderNetworkError,
)


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
