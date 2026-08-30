from dataclasses import FrozenInstanceError

import pytest

from coding_agent.approval import ApprovalRequest, request_approval


def test_approval_callback_can_accept_request() -> None:
    request = ApprovalRequest(operation="write_file", preview="create notes.txt")
    received: list[ApprovalRequest] = []

    def approve(candidate: ApprovalRequest) -> bool:
        received.append(candidate)
        return True

    assert request_approval(request, approve) is True
    assert received == [request]


def test_approval_callback_can_reject_request() -> None:
    request = ApprovalRequest(operation="edit_file", preview="replace one line")

    assert request_approval(request, lambda _: False) is False


def test_missing_approval_callback_defaults_to_rejection() -> None:
    request = ApprovalRequest(operation="write_file", preview="create notes.txt")

    assert request_approval(request) is False


def test_approval_request_operation_and_preview_are_immutable() -> None:
    request = ApprovalRequest(operation="write_file", preview="create notes.txt")

    with pytest.raises(FrozenInstanceError):
        request.operation = "edit_file"
    with pytest.raises(FrozenInstanceError):
        request.preview = "different content"
