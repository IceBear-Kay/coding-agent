from pathlib import Path

from coding_agent import (
    AgentLoop,
    AgentSession,
    FakeProvider,
    Message,
    ModelResponse,
    Usage,
    Workspace,
)
from coding_agent.compaction import COMPACTION_MARKER, compact_history
from coding_agent.session_store import SessionStore


def _history(count: int) -> list[Message]:
    messages: list[Message] = []
    for index in range(count):
        messages.extend(
            [
                Message(role="user", content=f"task {index} " + "constraint " * 20),
                Message(role="assistant", content=f"answer {index} " + "evidence " * 20),
            ]
        )
    return messages


def test_compact_history_summarizes_old_prefix_and_preserves_sources() -> None:
    history = _history(4)
    provider = FakeProvider(
        [
            ModelResponse(
                text=(
                    "目标与约束：保留只读边界；关键决定：保留只读；"
                    "已完成事项及证据：任务 0、1；待办：无；文件路径：无"
                ),
                finish_reason="stop",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            )
        ]
    )

    result = compact_history(provider, history)

    assert result.success
    assert result.record is not None
    assert result.record.covered_task_count == 2
    assert len(provider.requests) == 1
    assert provider.requests[0][1] == []
    assert provider.max_tokens == [4096]
    assert all(message in history for message in history)
    assert result.after_bytes < result.before_bytes


def test_compact_history_does_not_call_provider_without_three_tasks() -> None:
    provider = FakeProvider([])
    result = compact_history(provider, _history(2))
    assert not result.success
    assert result.reason == "nothing_to_compact"
    assert provider.requests == []


def test_session_uses_summary_only_in_request_view_and_persists_it(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="a " + "evidence " * 20, finish_reason="stop"),
            ModelResponse(text="b " + "evidence " * 20, finish_reason="stop"),
            ModelResponse(text="c " + "evidence " * 20, finish_reason="stop"),
            ModelResponse(
                text="目标：继续任务；关键决定：保留边界；已完成及证据：历史已读取；"
                "待办：无；文件路径：无",
                finish_reason="stop",
            ),
            ModelResponse(text="d", finish_reason="stop"),
        ]
    )
    session = AgentSession(AgentLoop(provider, Workspace(tmp_path), context_policy="compact"))
    for task in ("one", "two", "three"):
        assert session.run(task).stop_reason == "completed"

    compacted = session.compact()
    assert compacted.success
    assert session.state.compaction is not None
    assert len(session.state.messages) == 6
    final = session.run("four")
    assert final.stop_reason == "completed"
    assert any(
        message.content and COMPACTION_MARKER in message.content
        for message in provider.requests[-1][0]
    )
    assert all(message.content != "summary" for message in session.state.messages)


def test_persistent_session_round_trip_keeps_compaction_record(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    provider = FakeProvider(
        [
            ModelResponse(text="one " + "evidence " * 20, finish_reason="stop"),
            ModelResponse(text="two " + "evidence " * 20, finish_reason="stop"),
            ModelResponse(text="three " + "evidence " * 20, finish_reason="stop"),
            ModelResponse(
                text="目标：x；关键决定：y；已完成及证据：z；待办：无；文件路径：无",
                finish_reason="stop",
            ),
        ]
    )
    loop = AgentLoop(provider, Workspace(tmp_path), context_policy="compact")
    session = AgentSession.create(loop, store, "compact_1")
    for task in ("one", "two", "three"):
        session.run(task)
    assert session.compact().success
    session.close()

    resumed = AgentSession.resume(
        AgentLoop(FakeProvider([]), Workspace(tmp_path), context_policy="compact"),
        store,
        "compact_1",
    )
    assert resumed.state.compaction is not None
    assert resumed.state.compaction.summary.startswith("目标：")
    resumed.close()


def test_auto_compaction_runs_once_before_new_task_when_budget_reaches_threshold(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        [
            ModelResponse(text="a" * 500, finish_reason="stop"),
            ModelResponse(text="b" * 500, finish_reason="stop"),
            ModelResponse(text="c" * 500, finish_reason="stop"),
            ModelResponse(
                text="目标：x；关键决定：y；已完成及证据：z；待办：无；文件路径：无",
                finish_reason="stop",
            ),
            ModelResponse(text="d", finish_reason="stop"),
            ModelResponse(text="e", finish_reason="stop"),
        ]
    )
    session = AgentSession(
        AgentLoop(
            provider,
            Workspace(tmp_path),
            context_policy="compact",
            max_context_bytes=3500,
            max_context_tokens=1000,
        )
    )
    for task in ("one", "two", "three"):
        session.run(task)
    result = session.run("four")
    assert result.stop_reason == "completed"
    assert session.compaction_stats.provider_attempts == 1
    assert any(
        message.content and COMPACTION_MARKER in message.content
        for message in provider.requests[-1][0]
    )
    assert session.run("five").stop_reason == "completed"
    assert session.compaction_stats.provider_attempts == 1


def test_compaction_compares_against_existing_compressed_view() -> None:
    history = _history(7)
    first_provider = FakeProvider(
        [
            ModelResponse(
                text="目标：x；关键决定：y；已完成及证据：z；待办：无；文件路径：无",
                finish_reason="stop",
            )
        ]
    )
    first = compact_history(first_provider, history)
    assert first.success and first.record is not None
    history.extend(
        [
            Message(role="user", content="new task"),
            Message(role="assistant", content="new answer"),
        ]
    )
    second_provider = FakeProvider(
        [
            ModelResponse(
                text=(
                    "目标与约束：" + "很长" * 2000 + "；关键决定：x；"
                    "已完成事项及证据：y；待办：无；文件路径：无"
                ),
                finish_reason="stop",
            )
        ]
    )
    second = compact_history(second_provider, history, previous=first.record)
    assert not second.success
    assert second.reason == "summary_not_smaller"
