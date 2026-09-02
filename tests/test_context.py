import copy
import json

import pytest

from coding_agent.context import (
    DEFAULT_MAX_CONTEXT_BYTES,
    DEFAULT_MAX_CONTEXT_TOKENS,
    ContextBudget,
    ContextHistoryError,
    ContextSerializationError,
    check_context_budget,
    estimate_context_tokens,
    measure_context_bytes,
    select_context,
    serialize_context,
)
from coding_agent.models import Message, ToolCall


def test_empty_context_uses_compact_utf8_json() -> None:
    serialized = serialize_context([], [])

    assert serialized == b'{"messages":[],"tools":[]}'
    assert measure_context_bytes([], []) == len(serialized)
    assert check_context_budget([], []).within_budget
    assert DEFAULT_MAX_CONTEXT_BYTES == 8_388_608
    assert DEFAULT_MAX_CONTEXT_TOKENS == 524_288


def test_estimated_tokens_include_unicode_reasoning_tools_and_are_monotonic() -> None:
    base = [Message(role="user", content="读取文件")]
    with_reasoning = [
        Message(role="user", content="读取文件"),
        Message(role="assistant", content="答案", reasoning_content="检查路径"),
    ]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    assert estimate_context_tokens(base, []) >= 1
    assert estimate_context_tokens(with_reasoning, tools) > estimate_context_tokens(base, [])


def test_context_budget_can_check_estimated_token_limit() -> None:
    messages = [Message(role="user", content="a longer request")]
    used = estimate_context_tokens(messages, [])

    assert check_context_budget(messages, [], max_context_tokens=used).within_budget
    assert check_context_budget(messages, [], max_context_tokens=used - 1).exceeded


def test_context_serialization_is_deterministic_and_does_not_mutate_inputs() -> None:
    messages = [
        Message(
            role="assistant",
            content=None,
            reasoning_content="先读取文件。",
            tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "notes.txt"})],
        ),
        Message(role="tool", content="文件内容", tool_call_id="call_1"),
    ]
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    messages_before = copy.deepcopy(messages)
    tools_before = copy.deepcopy(tools)

    first = serialize_context(messages, tools)
    second = serialize_context(messages, tools)

    assert first == second
    assert json.loads(first) == {
        "messages": [
            {
                "reasoning_content": "先读取文件。",
                "role": "assistant",
                "tool_calls": [
                    {
                        "arguments": {"path": "notes.txt"},
                        "id": "call_1",
                        "name": "read_file",
                    }
                ],
            },
            {"content": "文件内容", "role": "tool", "tool_call_id": "call_1"},
        ],
        "tools": tools,
    }
    assert messages == messages_before
    assert tools == tools_before


def test_utf8_multibyte_text_is_counted_as_encoded_bytes() -> None:
    ascii_size = measure_context_bytes([Message(role="user", content="a")], [])
    chinese_size = measure_context_bytes([Message(role="user", content="中")], [])
    emoji_size = measure_context_bytes([Message(role="user", content="😀")], [])

    assert chinese_size == ascii_size + 2
    assert emoji_size == ascii_size + 3


def test_tool_schema_growth_is_included_in_measurement() -> None:
    messages = [Message(role="user", content="Inspect")]
    small_tools = [{"name": "read_file"}]
    large_tools = [{"name": "read_file", "description": "x" * 100}]

    assert measure_context_bytes(messages, large_tools) > measure_context_bytes(
        messages, small_tools
    )


def test_reasoning_content_tool_arguments_and_tool_result_are_counted() -> None:
    base = [Message(role="user", content="Inspect")]
    with_reasoning = [
        Message(role="assistant", content=None, reasoning_content="reasoning"),
    ]
    with_tool_call = [
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "a.txt"})],
        )
    ]
    with_tool_result = [
        Message(role="tool", content="result", tool_call_id="call_1"),
    ]

    assert measure_context_bytes(base + with_reasoning, []) > measure_context_bytes(base, [])
    assert measure_context_bytes(base + with_tool_call, []) > measure_context_bytes(base, [])
    assert measure_context_bytes(base + with_tool_result, []) > measure_context_bytes(base, [])


def test_budget_allows_exact_boundary_and_rejects_one_byte_over() -> None:
    messages = [Message(role="user", content="boundary")]
    used = measure_context_bytes(messages, [])

    exact = check_context_budget(messages, [], max_context_bytes=used)
    over = check_context_budget(messages, [], max_context_bytes=used - 1)

    assert exact.used_bytes == exact.max_bytes == used
    assert exact.within_budget is True
    assert exact.exceeded is False
    assert over.used_bytes == used
    assert over.exceeded is True


@pytest.mark.parametrize("value", [0, -1])
def test_budget_rejects_non_positive_limits(value: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        ContextBudget(value)


@pytest.mark.parametrize("value", [True, 1.5, "256"])
def test_budget_rejects_non_integer_limits(value: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        ContextBudget(value)  # type: ignore[arg-type]


def test_serialization_failure_does_not_expose_context_data() -> None:
    private_value = object()

    with pytest.raises(ContextSerializationError, match="context serialization failed") as exc_info:
        serialize_context([], [{"private": private_value}])

    assert str(exc_info.value) == "context serialization failed"
    assert "private" not in str(exc_info.value)


def test_select_context_stop_keeps_history_and_reports_budget() -> None:
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="old task"),
        Message(
            role="assistant",
            content=None,
            reasoning_content="reasoning",
            tool_calls=[ToolCall(id="call_old", name="read_file", arguments={"path": "old.txt"})],
        ),
        Message(role="tool", content="old result", tool_call_id="call_old"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="current task"),
    ]
    used = measure_context_bytes(messages, [])

    result = select_context(
        messages,
        [],
        current_task_start=5,
        max_context_bytes=used - 1,
        policy="stop",
    )

    assert result.messages == tuple(messages)
    assert result.used_bytes == used
    assert result.within_budget is False
    assert result.removed_task_count == 0


def test_select_context_trim_removes_oldest_complete_task_as_one_group() -> None:
    old_call = ToolCall(id="call_old", name="read_file", arguments={"path": "old.txt"})
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="old task"),
        Message(
            role="assistant",
            content=None,
            reasoning_content="reasoning",
            tool_calls=[old_call],
        ),
        Message(role="tool", content="old result", tool_call_id="call_old"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="current task"),
    ]
    current_only = [messages[0], messages[5]]
    budget = measure_context_bytes(current_only, [])
    original = copy.deepcopy(messages)

    result = select_context(
        messages,
        [],
        current_task_start=5,
        max_context_bytes=budget,
        policy="trim",
    )

    assert result.within_budget is True
    assert result.trimmed is True
    assert result.removed_task_count == 1
    assert result.messages == tuple(current_only)
    assert messages == original
    assert result.messages[1] is not messages[5]


def test_select_context_trim_removes_multiple_tasks_oldest_first() -> None:
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="oldest"),
        Message(role="assistant", content="answer one"),
        Message(role="user", content="middle"),
        Message(role="assistant", content="answer two"),
        Message(role="user", content="current"),
    ]
    budget = measure_context_bytes([messages[0], messages[5]], [])

    result = select_context(
        messages,
        [],
        current_task_start=5,
        max_context_bytes=budget,
        policy="trim",
    )

    assert result.messages == (messages[0], messages[5])
    assert result.removed_task_count == 2


def test_select_context_trim_keeps_current_task_even_when_it_cannot_fit() -> None:
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="old"),
        Message(role="assistant", content="answer"),
        Message(role="user", content="current task"),
    ]
    current_only_size = measure_context_bytes([messages[0], messages[3]], [])

    result = select_context(
        messages,
        [],
        current_task_start=3,
        max_context_bytes=current_only_size - 1,
        policy="trim",
    )

    assert result.messages == (messages[0], messages[3])
    assert result.within_budget is False
    assert result.removed_task_count == 1


@pytest.mark.parametrize("policy", ["invalid", "TRIM"])
def test_select_context_rejects_unknown_policy(policy: str) -> None:
    with pytest.raises(ValueError, match="context policy"):
        select_context(
            [Message(role="user", content="task")],
            [],
            current_task_start=0,
            policy=policy,  # type: ignore[arg-type]
        )


def test_select_context_rejects_incomplete_historical_task_before_trimming() -> None:
    messages = [
        Message(role="user", content="private old task"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_old", name="read_file", arguments={"path": "old.txt"})],
        ),
        Message(role="user", content="current task"),
    ]

    with pytest.raises(ContextHistoryError):
        select_context(messages, [], current_task_start=2, max_context_bytes=1, policy="trim")


def test_select_context_rejects_mismatched_tool_result() -> None:
    messages = [
        Message(role="user", content="private old task"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_same", name="read_file", arguments={"path": "old.txt"})],
        ),
        Message(role="tool", content="result", tool_call_id="wrong"),
        Message(role="user", content="current task"),
    ]

    with pytest.raises(ContextHistoryError):
        select_context(messages, [], current_task_start=3, policy="trim")


def test_select_context_rejects_missing_tool_result() -> None:
    messages = [
        Message(role="user", content="private old task"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_same", name="read_file", arguments={"path": "old.txt"})],
        ),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="current task"),
    ]

    with pytest.raises(ContextHistoryError):
        select_context(messages, [], current_task_start=3, policy="trim")


def test_select_context_allows_tool_call_id_reuse_across_tasks() -> None:
    messages = [
        Message(role="user", content="old task"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_same", name="read_file", arguments={"path": "old.txt"})],
        ),
        Message(role="tool", content="old result", tool_call_id="call_same"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="another old task"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_same", name="read_file", arguments={"path": "new.txt"})],
        ),
        Message(role="tool", content="new result", tool_call_id="call_same"),
        Message(role="assistant", content="new answer"),
        Message(role="user", content="current task"),
    ]
    current_only = [messages[8]]
    budget = measure_context_bytes(current_only, [])

    result = select_context(
        messages,
        [],
        current_task_start=8,
        max_context_bytes=budget,
        policy="trim",
    )

    assert result.messages == tuple(current_only)
    assert result.removed_task_count == 2


def test_select_context_trim_applies_token_budget_independently_of_bytes() -> None:
    messages = [
        Message(role="user", content="old task"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="current task"),
    ]
    current_tokens = estimate_context_tokens([messages[-1]], [])

    result = select_context(
        messages,
        [],
        current_task_start=2,
        max_context_bytes=8_388_608,
        max_context_tokens=current_tokens,
        policy="trim",
    )

    assert result.messages == (messages[-1],)
    assert result.used_tokens <= current_tokens
    assert result.removed_task_count == 1
