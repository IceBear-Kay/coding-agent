import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from coding_agent import (
    AgentLoop,
    AgentSession,
    CommandLimits,
    FakeProvider,
    Message,
    ModelResponse,
    ToolCall,
    Usage,
    Workspace,
    create_read_only_registry,
    create_workspace_registry,
    measure_context_bytes,
)
from coding_agent import session_store as session_store_module
from coding_agent.config import ProviderConfig
from coding_agent.errors import ProviderNetworkError
from coding_agent.provider import OpenAICompatibleProvider
from coding_agent.session import SESSION_LOCK_ERROR_STOP_REASON, SESSION_SAVE_ERROR_STOP_REASON
from coding_agent.session_store import SessionConflictError, SessionStore, SessionStoreError


def test_session_preserves_completed_history_and_resets_task_state(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="First answer", finish_reason="stop"),
            ModelResponse(text="Second answer", finish_reason="stop"),
        ]
    )
    session = AgentSession(
        AgentLoop(
            provider,
            Workspace(tmp_path),
            max_steps=1,
            system_prompt="System instructions",
        )
    )

    first = session.run("First task")
    second = session.run("Second task")

    assert first.state is not second.state
    assert first.state.step_count == second.state.step_count == 1
    assert [message.role for message in provider.requests[0][0]] == ["system", "user"]
    assert [message.role for message in provider.requests[1][0]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert sum(message.role == "system" for message in provider.requests[1][0]) == 1
    assert provider.requests[1][0][-1].content == "Second task"
    assert session.state.messages == second.state.messages
    assert session.state.messages is not second.state.messages


def test_session_task_stats_are_reset_for_each_task(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                text="First answer",
                finish_reason="stop",
                usage=Usage(input_tokens=3, output_tokens=1, total_tokens=4),
            ),
            ModelResponse(
                text="Second answer",
                finish_reason="stop",
                usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
            ),
        ]
    )
    session = AgentSession(AgentLoop(provider, Workspace(tmp_path), max_steps=2))

    first = session.run("First task")
    second = session.run("Second task")

    assert first.stats.provider_attempts == second.stats.provider_attempts == 1
    assert first.stats.total_tokens == 4
    assert second.stats.total_tokens == 7


def test_session_preserves_tool_protocol_and_reasoning_across_tasks(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("remember this", encoding="utf-8")
    provider = FakeProvider(
        [
            ModelResponse(
                reasoning_content="I need the file.",
                tool_calls=[
                    ToolCall(
                        id="call_notes",
                        name="read_file",
                        arguments={"path": "notes.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(
                text="I read the note.",
                reasoning_content="The file is available.",
                finish_reason="stop",
            ),
            ModelResponse(text="The note said remember this.", finish_reason="stop"),
        ]
    )
    session = AgentSession(AgentLoop(provider, Workspace(tmp_path), max_steps=2))

    first = session.run("Read notes.txt")
    second = session.run("What did it say?")

    assert first.state.step_count == 2
    assert second.state.step_count == 1
    second_request = provider.requests[2][0]
    assistant_tool_message = next(message for message in second_request if message.tool_calls)
    tool_result_message = next(message for message in second_request if message.role == "tool")
    assert assistant_tool_message.reasoning_content == "I need the file."
    assert assistant_tool_message.tool_calls[0].id == "call_notes"
    assert tool_result_message.tool_call_id == "call_notes"
    assert tool_result_message.content == "remember this"


def test_session_does_not_commit_an_abnormally_stopped_task(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="Committed answer", finish_reason="stop"),
            ModelResponse(
                text=None,
                reasoning_content="The response was truncated.",
                finish_reason="length",
            ),
        ]
    )
    session = AgentSession(AgentLoop(provider, Workspace(tmp_path)))

    completed = session.run("Complete this")
    committed_history = [message.model_copy(deep=True) for message in session.state.messages]
    stopped = session.run("This will be truncated")

    assert completed.stop_reason == "completed"
    assert stopped.stop_reason == "length"
    assert session.state.messages == committed_history
    assert Message(role="user", content="This will be truncated") in stopped.state.messages
    assert Message(role="user", content="This will be truncated") not in session.state.messages


def test_session_max_steps_is_scoped_to_each_task(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_limited",
                        name="list_files",
                        arguments={},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="第二项独立完成。", finish_reason="stop"),
        ]
    )
    session = AgentSession(AgentLoop(provider, Workspace(tmp_path), max_steps=1))

    limited = session.run("第一项触发步数上限。")
    completed = session.run("第二项重新开始计数。")

    assert limited.stop_reason == "max_steps"
    assert completed.stop_reason == "completed"
    assert limited.state.step_count == completed.state.step_count == 1
    assert [message.role for message in provider.requests[1][0]] == ["user"]
    assert provider.requests[1][0][0].content == "第二项重新开始计数。"
    assert session.messages == completed.state.messages
    assert all(message.content != "第一项触发步数上限。" for message in session.messages)


def test_session_context_budget_applies_to_accumulated_history(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    registry = create_read_only_registry(workspace)
    first_task = "第一项"
    second_task = "第二项"
    third_task = "第三项"
    first_answer = "第一项完成"
    second_answer = "第二项完成"
    budget_messages = [
        Message(role="user", content=first_task),
        Message(role="assistant", content=first_answer),
        Message(role="user", content=second_task),
    ]
    max_context_bytes = measure_context_bytes(budget_messages, registry.schemas())
    provider = FakeProvider(
        [
            ModelResponse(text=first_answer, finish_reason="stop"),
            ModelResponse(text=second_answer, finish_reason="stop"),
            ModelResponse(text="不应请求", finish_reason="stop"),
        ]
    )
    session = AgentSession(
        AgentLoop(
            provider,
            workspace,
            registry=registry,
            max_steps=1,
            max_context_bytes=max_context_bytes,
        )
    )

    first = session.run(first_task)
    second = session.run(second_task)
    third = session.run(third_task)

    assert first.stop_reason == second.stop_reason == "completed"
    assert second.state.step_count == 1
    assert third.stop_reason == "completed"
    assert third.state.step_count == 1
    assert len(provider.requests) == 3
    assert session.messages == third.state.messages
    assert third_task in [message.content for message in session.messages]
    assert third.state.context_trimmed_tasks == 1


def test_session_retry_budget_resets_for_next_task(tmp_path: Path) -> None:
    class RetryProvider:
        def __init__(self) -> None:
            self.outcomes = [
                ProviderNetworkError("temporary one"),
                ModelResponse(text="第一项完成。", finish_reason="stop"),
                ProviderNetworkError("temporary two"),
                ModelResponse(text="第二项完成。", finish_reason="stop"),
            ]
            self.requests: list[list[Message]] = []

        def complete(self, messages, tool_schemas):
            self.requests.append([message.model_copy(deep=True) for message in messages])
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    provider = RetryProvider()
    session = AgentSession(AgentLoop(provider, Workspace(tmp_path), max_steps=2, max_retries=1))

    first = session.run("第一项任务。")
    second = session.run("第二项任务。")

    assert first.stop_reason == second.stop_reason == "completed"
    assert first.state.step_count == second.state.step_count == 2
    assert len(provider.requests) == 4
    assert provider.requests[2][-1].content == "第二项任务。"


def test_session_http_provider_sends_cross_task_tool_history(tmp_path: Path) -> None:
    request_bodies: list[dict[str, Any]] = []

    def tool_response(call_id: str, name: str, arguments: dict[str, Any], reasoning: str):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": reasoning,
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
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
        body = json.loads(request.content)
        request_bodies.append(body)
        if len(request_bodies) == 1:
            return tool_response(
                "call_http_create",
                "write_file",
                {"path": "http.txt", "content": "created over HTTP"},
                "创建文件。",
            )
        if len(request_bodies) == 2:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "第一项完成。"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        if len(request_bodies) == 3:
            return tool_response(
                "call_http_read",
                "read_file",
                {"path": "http.txt"},
                "读取已创建文件。",
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "第二项完成。"},
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
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        approval_callback=lambda _: True,
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(config, client)
        session = AgentSession(
            AgentLoop(
                provider,
                workspace,
                registry=registry,
                system_prompt="会话系统提示。",
                max_steps=2,
            )
        )
        first = session.run("创建 http.txt。")
        second = session.run("读取 http.txt。")

    assert first.stop_reason == second.stop_reason == "completed"
    assert (tmp_path / "http.txt").read_text(encoding="utf-8") == "created over HTTP"
    assert len(request_bodies) == 4
    second_task_messages = request_bodies[2]["messages"]
    assert [message["role"] for message in second_task_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert second_task_messages[2]["reasoning_content"] == "创建文件。"
    assert second_task_messages[2]["tool_calls"][0]["id"] == "call_http_create"
    assert second_task_messages[3] == {
        "role": "tool",
        "tool_call_id": "call_http_create",
        "content": '{"bytes_written":17,"path":"http.txt","status":"created"}',
    }
    assert second_task_messages[4]["content"] == "第一项完成。"
    follow_up_messages = request_bodies[3]["messages"]
    assert follow_up_messages[-2]["tool_calls"][0]["id"] == "call_http_read"
    assert follow_up_messages[-2]["reasoning_content"] == "读取已创建文件。"
    assert follow_up_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_http_read",
        "content": "created over HTTP",
    }


def test_session_runs_real_tools_across_tasks_with_independent_approvals(tmp_path: Path) -> None:
    approvals: list[str] = []
    workspace = Workspace(tmp_path)
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        approval_callback=lambda request: approvals.append(request.operation) or True,
    )
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_create",
                        name="write_file",
                        arguments={"path": "notes.txt", "content": "created in task one"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="文件已创建。", finish_reason="stop"),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_read",
                        name="read_file",
                        arguments={"path": "notes.txt"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="第二个任务读到了文件内容。", finish_reason="stop"),
        ]
    )
    session = AgentSession(AgentLoop(provider, workspace, registry=registry, max_steps=3))

    first = session.run("创建 notes.txt。")
    second = session.run("读取 notes.txt 并确认内容。")

    assert first.stop_reason == second.stop_reason == "completed"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "created in task one"
    assert approvals == ["write_file"]
    assert second.state.step_count == 2
    second_request = provider.requests[2][0]
    assert second_request[-1] == Message(role="user", content="读取 notes.txt 并确认内容。")
    assert [message.tool_call_id for message in second_request if message.role == "tool"] == [
        "call_create"
    ]
    follow_up_request = provider.requests[3][0]
    assert follow_up_request[-2].role == "assistant"
    assert follow_up_request[-2].tool_calls[0].id == "call_read"
    assert follow_up_request[-1].tool_call_id == "call_read"
    assert follow_up_request[-1].content == "created in task one"


def test_session_rejection_is_independent_for_each_task(tmp_path: Path) -> None:
    decisions = iter([True, False])
    approvals: list[str] = []
    workspace = Workspace(tmp_path)
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        approval_callback=lambda request: approvals.append(request.operation) or next(decisions),
    )
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_first",
                        name="write_file",
                        arguments={"path": "first.txt", "content": "approved"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="第一项已完成。", finish_reason="stop"),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_second",
                        name="write_file",
                        arguments={"path": "second.txt", "content": "must be denied"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="第二项被拒绝。", finish_reason="stop"),
        ]
    )
    session = AgentSession(AgentLoop(provider, workspace, registry=registry, max_steps=3))

    first = session.run("创建 first.txt。")
    second = session.run("创建 second.txt。")

    assert first.stop_reason == second.stop_reason == "completed"
    assert approvals == ["write_file", "write_file"]
    assert (tmp_path / "first.txt").read_text(encoding="utf-8") == "approved"
    assert not (tmp_path / "second.txt").exists()
    denied = next(
        message for message in second.state.messages if message.tool_call_id == "call_second"
    )
    assert '"status":"denied"' in denied.content


def test_session_provider_error_ends_task_without_committing_history(tmp_path: Path) -> None:
    from coding_agent.errors import ProviderAuthenticationError

    provider_error = ProviderAuthenticationError("invalid credentials")

    class ErrorProvider:
        requests: list[tuple[list[Message], list[dict[str, object]]]] = []

        def complete(self, messages, tool_schemas):
            self.requests.append((list(messages), list(tool_schemas)))
            raise provider_error

    error_provider = ErrorProvider()
    session = AgentSession(AgentLoop(error_provider, Workspace(tmp_path)))

    result = session.run("调用模型。")

    assert result.stop_reason == "fatal_error"
    assert result.error is provider_error
    assert session.messages == []
    assert len(error_provider.requests) == 1


def test_persistent_session_saves_only_completed_tasks(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    provider = FakeProvider(
        [
            ModelResponse(text="first answer", finish_reason="stop"),
            ModelResponse(
                text=None,
                reasoning_content="truncated",
                finish_reason="length",
            ),
        ]
    )
    session = AgentSession.create(AgentLoop(provider, Workspace(tmp_path)), store, "chat_1")

    completed = session.run("first task")
    archive_after_completed = store.load("chat_1", workspace_root=tmp_path)
    stopped = session.run("second task")

    assert completed.stop_reason == "completed"
    assert completed.error is None
    assert stopped.stop_reason == "length"
    assert archive_after_completed.messages == session.messages
    assert store.load("chat_1", workspace_root=tmp_path) == archive_after_completed
    assert Message(role="user", content="second task") in stopped.state.messages
    assert Message(role="user", content="second task") not in archive_after_completed.messages


def test_persistent_session_resume_does_not_replay_history(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    workspace = Workspace(tmp_path)
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        approval_callback=lambda _: True,
    )
    first_provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_create",
                        name="write_file",
                        arguments={"path": "note.txt", "content": "saved"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            ModelResponse(text="created", finish_reason="stop"),
        ]
    )
    first_session = AgentSession.create(
        AgentLoop(first_provider, workspace, registry=registry),
        store,
        "chat_1",
    )
    assert first_session.run("create note").stop_reason == "completed"
    first_session.close()

    second_provider = FakeProvider([ModelResponse(text="continued", finish_reason="stop")])
    resumed = AgentSession.resume(
        AgentLoop(
            second_provider, Workspace(tmp_path), registry=create_read_only_registry(workspace)
        ),
        store,
        "chat_1",
    )

    assert second_provider.requests == []
    continued = resumed.run("what was created?")

    assert continued.stop_reason == "completed"
    assert len(second_provider.requests) == 1
    request_messages = second_provider.requests[0][0]
    assert any(message.tool_call_id == "call_create" for message in request_messages)
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "saved"


def test_persistent_session_rejects_second_resume_while_first_is_active(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    first = AgentSession.create(
        AgentLoop(FakeProvider([]), Workspace(tmp_path)),
        store,
        "chat_1",
    )

    with pytest.raises(SessionConflictError, match="already in use"):
        AgentSession.resume(
            AgentLoop(FakeProvider([]), Workspace(tmp_path)),
            store,
            "chat_1",
        )
    first.close()


def test_closed_session_run_is_rejected_before_provider_or_tools(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    provider = FakeProvider([])
    first = AgentSession.create(AgentLoop(provider, Workspace(tmp_path)), store, "chat_1")
    first.close()

    result = first.run("不应执行")

    assert result.stop_reason == SESSION_LOCK_ERROR_STOP_REASON
    assert result.stats.stop_reason == result.stop_reason
    assert provider.requests == []


def test_closed_session_cannot_run_while_another_instance_holds_lock(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    first_provider = FakeProvider([])
    first = AgentSession.create(AgentLoop(first_provider, Workspace(tmp_path)), store, "chat_1")
    first.close()
    resumed_provider = FakeProvider([ModelResponse(text="继续", finish_reason="stop")])
    resumed = AgentSession.resume(AgentLoop(resumed_provider, Workspace(tmp_path)), store, "chat_1")

    result = first.run("不应执行")

    assert result.stop_reason == SESSION_LOCK_ERROR_STOP_REASON
    assert result.stats.stop_reason == result.stop_reason
    assert first_provider.requests == []
    assert resumed_provider.requests == []
    resumed.close()


def test_resume_keyboard_interrupt_releases_its_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.create("chat_1", tmp_path)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "load", interrupt)
    with pytest.raises(KeyboardInterrupt):
        AgentSession.resume(AgentLoop(FakeProvider([]), Workspace(tmp_path)), store, "chat_1")

    lease = store.acquire("chat_1")
    lease.release()


def test_persistent_resume_rebuilds_system_prompt_for_current_run(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    first_provider = FakeProvider([ModelResponse(text="old answer", finish_reason="stop")])
    first = AgentSession.create(
        AgentLoop(first_provider, Workspace(tmp_path), system_prompt="old rules"),
        store,
        "chat_1",
    )
    assert first.run("first task").stop_reason == "completed"
    first.close()

    second_provider = FakeProvider([ModelResponse(text="new answer", finish_reason="stop")])
    resumed = AgentSession.resume(
        AgentLoop(second_provider, Workspace(tmp_path), system_prompt="new rules"),
        store,
        "chat_1",
    )

    result = resumed.run("second task")

    assert result.stop_reason == "completed"
    request_messages = second_provider.requests[0][0]
    assert request_messages[0] == Message(role="system", content="new rules")
    assert sum(message.role == "system" for message in request_messages) == 1
    assert request_messages[-1] == Message(role="user", content="second task")


def test_persistent_session_save_failure_keeps_real_result_and_old_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = AgentSession.create(
        AgentLoop(
            FakeProvider([ModelResponse(text="answer", finish_reason="stop")]),
            Workspace(tmp_path),
        ),
        store,
        "chat_1",
    )
    original = store.load("chat_1", workspace_root=tmp_path)

    def fail_save(*args, **kwargs):
        raise SessionStoreError("simulated save failure")

    monkeypatch.setattr(store, "save", fail_save)
    result = session.run("task with real result")

    assert result.stop_reason == SESSION_SAVE_ERROR_STOP_REASON
    assert result.stats.stop_reason == result.stop_reason
    assert isinstance(result.error, SessionStoreError)
    assert result.answer == "answer"
    assert Message(role="user", content="task with real result") in session.messages
    assert store.load("chat_1", workspace_root=tmp_path) == original


def test_persistent_session_archive_io_failure_keeps_answer_and_old_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "sessions")
    session = AgentSession.create(
        AgentLoop(
            FakeProvider([ModelResponse(text="真实答案", finish_reason="stop")]),
            Workspace(tmp_path),
        ),
        store,
        "chat_1",
    )
    original = store.load("chat_1", workspace_root=tmp_path)

    def fail_fsync(_: int) -> None:
        raise OSError("simulated archive fsync failure")

    monkeypatch.setattr(session_store_module.os, "fsync", fail_fsync)
    result = session.run("执行并保存")

    assert result.stop_reason == SESSION_SAVE_ERROR_STOP_REASON
    assert result.stats.stop_reason == result.stop_reason
    assert result.answer == "真实答案"
    assert isinstance(result.error, SessionStoreError)
    assert store.load("chat_1", workspace_root=tmp_path) == original
    session.close()


def test_persistent_session_store_directory_is_protected_from_file_tools(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    workspace = Workspace(tmp_path)
    loop = AgentLoop(
        FakeProvider([]),
        workspace,
        registry=create_workspace_registry(
            workspace,
            allow_write=True,
            approval_callback=lambda _: True,
        ),
    )
    session = AgentSession.create(loop, store, "chat_1")

    read_result = session.loop.dispatcher.dispatch(
        ToolCall(id="read", name="read_file", arguments={"path": "sessions/chat_1.json"})
    )
    list_result = session.loop.dispatcher.dispatch(
        ToolCall(id="list", name="list_files", arguments={"path": "."})
    )
    write_result = session.loop.dispatcher.dispatch(
        ToolCall(
            id="write",
            name="write_file",
            arguments={"path": "sessions/new.json", "content": "blocked"},
        )
    )

    assert read_result.is_error is True
    assert "protected" in read_result.content.lower()
    assert "sessions/chat_1.json" not in list_result.content
    assert write_result.is_error is True
    assert not (tmp_path / "sessions" / "new.json").exists()


def test_session_interrupt_after_first_of_multiple_tools_keeps_completed_result(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    registry = create_workspace_registry(
        workspace,
        allow_write=True,
        approval_callback=lambda _: True,
        command_limits=CommandLimits(timeout_seconds=1),
    )
    provider = FakeProvider(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call_one",
                        name="write_file",
                        arguments={"path": "one.txt", "content": "done"},
                    ),
                    ToolCall(
                        id="call_two",
                        name="write_file",
                        arguments={"path": "two.txt", "content": "not run"},
                    ),
                ],
                finish_reason="tool_calls",
            )
        ]
    )
    events = []

    def interrupt_after_first_result(event) -> None:
        events.append(event)
        if event.kind == "tool_result":
            raise KeyboardInterrupt

    loop = AgentLoop(
        provider,
        workspace,
        registry=registry,
        event_callback=interrupt_after_first_result,
    )
    session = AgentSession(loop)

    result = session.run("创建两个文件。")

    assert result.stop_reason == "interrupted"
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "done"
    assert not (tmp_path / "two.txt").exists()
    assert [event.kind for event in events] == ["tool_call", "tool_result"]
    assert [
        message.tool_call_id for message in result.state.messages if message.role == "tool"
    ] == ["call_one"]
    assert session.messages == []
    assert len(provider.requests) == 1
