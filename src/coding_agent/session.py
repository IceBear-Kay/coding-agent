"""In-memory session coordination for consecutive completed tasks."""

from coding_agent.agent import COMPLETED_STOP_REASON, AgentLoop, AgentRunResult
from coding_agent.models import Message, SessionState


class AgentSession:
    """Keep one authoritative history while each task gets a fresh AgentState."""

    def __init__(self, loop: AgentLoop) -> None:
        self.loop = loop
        self.state = SessionState()

    @property
    def messages(self) -> list[Message]:
        """Expose the committed history for inspection without changing ownership."""
        return self.state.messages

    def run(self, task: str) -> AgentRunResult:
        """Run a task and commit its complete history only after normal completion."""
        result = self.loop.run(task, history=self.state.messages)
        if result.stop_reason == COMPLETED_STOP_REASON:
            self.state.messages = [
                message.model_copy(deep=True) for message in result.state.messages
            ]
        return result

    def clear(self) -> None:
        """Discard committed messages while retaining the loop configuration."""
        self.state.messages.clear()


__all__ = ["AgentSession"]
