import pytest
from pydantic import ValidationError

from coding_agent.models import Message, ModelResponse, ToolCall, Usage


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
def test_message_accepts_supported_text_roles(role: str) -> None:
    message = Message(role=role, content="Hello")

    assert message.model_dump() == {"role": role, "content": "Hello"}


def test_message_rejects_unsupported_role() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Message.model_validate({"role": "tool", "content": "Result"})

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

    assert exc_info.value.errors()[0]["loc"] == ("content",)


def test_message_rejects_non_string_content() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Message.model_validate({"role": "user", "content": 123})

    assert exc_info.value.errors()[0]["loc"] == ("content",)


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


def test_usage_rejects_negative_token_counts() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Usage(input_tokens=-1, output_tokens=3, total_tokens=2)

    assert exc_info.value.errors()[0]["loc"] == ("input_tokens",)
