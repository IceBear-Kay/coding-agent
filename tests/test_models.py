import pytest
from pydantic import ValidationError

from coding_agent.models import Message


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
