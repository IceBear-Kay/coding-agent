from pathlib import Path

import pytest
from pydantic import ValidationError

from coding_agent.models import AgentState, Message, ModelResponse, ToolCall, ToolResult, Usage


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_message_accepts_supported_text_roles(role: str) -> None:
    message = Message(role=role, content="Hello")

    assert message.model_dump() == {"role": role, "content": "Hello"}


def test_message_rejects_unsupported_role() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Message.model_validate({"role": "developer", "content": "Result"})

    assert exc_info.value.errors()[0]["loc"] == ("role",)


def test_message_requires_role_and_content() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Message.model_validate({})

    assert {error["loc"] for error in exc_info.value.errors()} == {
        ("role",),
        ("content",),
    }


def test_message_rejects_empty_content() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Message(role="user", content="")

    assert exc_info.value.errors()[0]["loc"] == ()


def test_message_rejects_non_string_content() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Message.model_validate({"role": "user", "content": 123})

    assert exc_info.value.errors()[0]["loc"] == ("content",)


def test_assistant_message_preserves_reasoning_and_tool_calls() -> None:
    call = ToolCall(
        id="call_1",
        name="read_file",
        arguments={"path": "README.md"},
    )

    message = Message(
        role="assistant",
        content=None,
        reasoning_content="I should inspect the README first.",
        tool_calls=[call],
    )

    assert message.content is None
    assert message.reasoning_content == "I should inspect the README first."
    assert message.tool_calls == [call]


@pytest.mark.parametrize("content", ["", None])
def test_assistant_message_allows_empty_or_null_content(content: str | None) -> None:
    message = Message(role="assistant", content=content)

    assert message.content == content


def test_tool_message_preserves_call_id_and_result() -> None:
    message = Message(
        role="tool",
        content="README contents",
        tool_call_id="call_1",
    )

    assert message.model_dump() == {
        "role": "tool",
        "content": "README contents",
        "tool_call_id": "call_1",
    }


def test_tool_message_requires_call_id() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        Message(role="tool", content="README contents")


def test_non_assistant_messages_reject_reasoning_and_tool_calls() -> None:
    with pytest.raises(ValidationError, match="reasoning_content"):
        Message(
            role="user",
            content="Inspect the project",
            reasoning_content="Hidden reasoning",
        )

    with pytest.raises(ValidationError, match="tool_calls"):
        Message(
            role="user",
            content="Inspect the project",
            tool_calls=[ToolCall(id="call_1", name="read_file", arguments={})],
        )


def test_agent_state_has_safe_runtime_defaults(tmp_path: Path) -> None:
    state = AgentState(workspace_root=tmp_path, max_steps=5)

    assert state.workspace_root == tmp_path
    assert state.max_steps == 5
    assert state.messages == []
    assert state.step_count == 0
    assert state.stop_reason is None


def test_agent_state_builds_nested_messages(tmp_path: Path) -> None:
    state = AgentState.model_validate(
        {
            "workspace_root": tmp_path,
            "max_steps": 5,
            "messages": [{"role": "user", "content": "Inspect the project"}],
            "step_count": 1,
            "stop_reason": "completed",
        }
    )

    assert state.messages == [Message(role="user", content="Inspect the project")]
    assert state.step_count == 1
    assert state.stop_reason == "completed"


def test_agent_state_accepts_assistant_and_tool_history(tmp_path: Path) -> None:
    history = [
        Message(role="user", content="Read README.md"),
        Message(
            role="assistant",
            content=None,
            reasoning_content="I will call read_file.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="read_file",
                    arguments={"path": "README.md"},
                )
            ],
        ),
        Message(role="tool", content="README contents", tool_call_id="call_1"),
    ]

    state = AgentState(workspace_root=tmp_path, max_steps=5, messages=history)

    assert state.messages == history


def test_agent_states_do_not_share_message_lists(tmp_path: Path) -> None:
    first = AgentState(workspace_root=tmp_path, max_steps=5)
    second = AgentState(workspace_root=tmp_path, max_steps=5)

    first.messages.append(Message(role="user", content="First task"))

    assert len(first.messages) == 1
    assert second.messages == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("step_count", -1), ("max_steps", 0)],
)
def test_agent_state_rejects_invalid_step_counts(
    field: str,
    value: int,
    tmp_path: Path,
) -> None:
    data = {"workspace_root": tmp_path, "max_steps": 5, field: value}

    with pytest.raises(ValidationError) as exc_info:
        AgentState.model_validate(data)

    assert exc_info.value.errors()[0]["loc"] == (field,)


def test_agent_state_requires_workspace_and_step_limit() -> None:
    with pytest.raises(ValidationError) as exc_info:
        AgentState.model_validate({})

    assert {error["loc"] for error in exc_info.value.errors()} == {
        ("workspace_root",),
        ("max_steps",),
    }


def test_agent_state_rejects_empty_stop_reason(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as exc_info:
        AgentState(workspace_root=tmp_path, max_steps=5, stop_reason="")

    assert exc_info.value.errors()[0]["loc"] == ("stop_reason",)


def test_model_response_stores_text_and_metadata() -> None:
    response = ModelResponse(
        text="Done",
        usage=Usage(input_tokens=12, output_tokens=3, total_tokens=15),
        finish_reason="stop",
    )

    assert response.text == "Done"
    assert response.tool_calls == []
    assert response.usage is not None
    assert response.usage.total_tokens == 15
    assert response.finish_reason == "stop"


def test_model_response_builds_nested_tool_calls() -> None:
    response = ModelResponse.model_validate(
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                }
            ],
            "finish_reason": "tool_calls",
        }
    )

    assert response.tool_calls == [
        ToolCall(
            id="call_1",
            name="read_file",
            arguments={"path": "README.md"},
        )
    ]


def test_model_response_preserves_multiple_tool_call_order() -> None:
    calls = [
        ToolCall(id="call_1", name="list_files", arguments={"path": "."}),
        ToolCall(id="call_2", name="read_file", arguments={"path": "README.md"}),
    ]

    response = ModelResponse(tool_calls=calls)

    assert [call.id for call in response.tool_calls] == ["call_1", "call_2"]


@pytest.mark.parametrize("field", ["id", "name"])
def test_tool_call_rejects_empty_identifiers(field: str) -> None:
    data = {"id": "call_1", "name": "read_file", "arguments": {}}
    data[field] = ""

    with pytest.raises(ValidationError) as exc_info:
        ToolCall.model_validate(data)

    assert exc_info.value.errors()[0]["loc"] == (field,)


def test_tool_call_requires_object_arguments() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolCall.model_validate({"id": "call_1", "name": "read_file", "arguments": "{}"})

    assert exc_info.value.errors()[0]["loc"] == ("arguments",)


def test_tool_result_defaults_to_success() -> None:
    result = ToolResult(tool_call_id="call_1", content="File contents")

    assert result.model_dump() == {
        "tool_call_id": "call_1",
        "content": "File contents",
        "is_error": False,
    }


def test_tool_result_represents_an_error() -> None:
    result = ToolResult(
        tool_call_id="call_1",
        content="File not found",
        is_error=True,
    )

    assert result.tool_call_id == "call_1"
    assert result.content == "File not found"
    assert result.is_error is True


def test_tool_result_rejects_empty_call_id() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolResult(tool_call_id="", content="File contents")

    assert exc_info.value.errors()[0]["loc"] == ("tool_call_id",)


def test_tool_result_requires_content() -> None:
    with pytest.raises(ValidationError) as exc_info:
        ToolResult.model_validate({"tool_call_id": "call_1"})

    assert exc_info.value.errors()[0]["loc"] == ("content",)


def test_usage_rejects_negative_token_counts() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Usage(input_tokens=-1, output_tokens=3, total_tokens=2)

    assert exc_info.value.errors()[0]["loc"] == ("input_tokens",)
