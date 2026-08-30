"""Approval contract for local operations with side effects."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Describe one operation and the exact preview presented for approval."""

    operation: str
    preview: str

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("Approval operation must not be empty")
        if not isinstance(self.preview, str):
            raise TypeError("Approval preview must be text")


ApprovalCallback = Callable[[ApprovalRequest], bool]


def request_approval(
    request: ApprovalRequest,
    callback: ApprovalCallback | None = None,
) -> bool:
    """Return approval only when an injected callback explicitly grants it."""
    if callback is None:
        return False
    return callback(request) is True
