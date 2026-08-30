"""The minimal provider-tool agent loop."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from coding_agent.errors import FatalProviderError, ProviderError, TransientProviderError
from coding_agent.models import AgentState, Message, ModelResponse
from coding_agent.provider import ModelProvider
from coding_agent.tools import (
    ToolDispatcher,
    ToolRegistry,
    Workspace,
    create_read_only_registry,
)

COMPLETED_STOP_REASON = "completed"
MAX_STEPS_STOP_REASON = "max_steps"
INTERRUPTED_STOP_REASON = "interrupted"
FATAL_ERROR_STOP_REASON = "fatal_error"
PROVIDER_ERROR_STOP_REASON = "provider_error"
TRANSIENT_PROVIDER_ERROR_STOP_REASON = "transient_provider_error"
DEFAULT_MAX_STEPS = 8
NORMAL_FINISH_REASONS = frozenset({None, "stop", "completed"})
NON_NORMAL_FINISH_REASONS = frozenset({"length", "content_filter", "insufficient_system_resource"})


@dataclass(frozen=True)
class AgentRunResult:
    """The final answer together with the state produced by one task run."""

    answer: str | None
    state: AgentState
    error: BaseException | None = None

    @property
    def final_answer(self) -> str | None:
        """Expose the answer under a descriptive name for callers."""
        return self.answer

    @property
    def stop_reason(self) -> str | None:
        """Expose the terminal state without requiring callers to unpack it."""
        return self.state.stop_reason


class AgentLoop:
    """Run one task through a provider and a local tool dispatcher."""

    def __init__(
        self,
        provider: ModelProvider,
        workspace: Workspace,
        *,
        registry: ToolRegistry | None = None,
        dispatcher: ToolDispatcher | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must not be negative")

        self.provider = provider
        self.workspace = workspace
        self.registry = registry if registry is not None else create_read_only_registry(workspace)
        self.dispatcher = dispatcher if dispatcher is not None else ToolDispatcher(self.registry)
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.sleep = sleep
        self.state: AgentState | None = None

    def run(self, task: str) -> AgentRunResult:
        """Execute one task and return its answer and complete conversation state."""
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        state = AgentState(
            workspace_root=Path(self.workspace.root),
            max_steps=self.max_steps,
        )
        state.messages.append(Message(role="user", content=task))
        self.state = state
        tool_schemas = self.registry.schemas()
        retry_count = 0

        try:
            while state.step_count < state.max_steps:
                # Count every provider attempt, including transient failures, so retries
                # cannot exceed the caller's global invocation budget.
                state.step_count += 1
                try:
                    response = self.provider.complete(state.messages, tool_schemas)
                except TransientProviderError as exc:
                    if retry_count >= self.max_retries or state.step_count >= state.max_steps:
                        state.stop_reason = TRANSIENT_PROVIDER_ERROR_STOP_REASON
                        return AgentRunResult(answer=None, state=state, error=exc)
                    retry_count += 1
                    delay = self.retry_delay_seconds * (2 ** (retry_count - 1))
                    if delay:
                        self.sleep(delay)
                    continue
                except FatalProviderError as exc:
                    state.stop_reason = FATAL_ERROR_STOP_REASON
                    return AgentRunResult(answer=None, state=state, error=exc)
                except ProviderError as exc:
                    state.stop_reason = PROVIDER_ERROR_STOP_REASON
                    return AgentRunResult(answer=None, state=state, error=exc)
                except Exception as exc:
                    state.stop_reason = PROVIDER_ERROR_STOP_REASON
                    return AgentRunResult(answer=None, state=state, error=exc)

                retry_count = 0
                self._append_assistant_message(state, response)

                if response.finish_reason in NON_NORMAL_FINISH_REASONS:
                    state.stop_reason = response.finish_reason
                    return AgentRunResult(answer=response.text, state=state)

                if not response.tool_calls:
                    state.stop_reason = self._finish_stop_reason(response)
                    return AgentRunResult(answer=response.text, state=state)

                for tool_call in response.tool_calls:
                    tool_result = self.dispatcher.dispatch(tool_call)
                    state.messages.append(
                        Message(
                            role="tool",
                            tool_call_id=tool_call.id,
                            content=tool_result.content,
                        )
                    )

                if state.step_count >= state.max_steps:
                    state.stop_reason = MAX_STEPS_STOP_REASON
                    return AgentRunResult(answer=None, state=state)
        except KeyboardInterrupt as exc:
            state.stop_reason = INTERRUPTED_STOP_REASON
            return AgentRunResult(answer=None, state=state, error=exc)

        # The loop always returns from the body because max_steps is positive.
        raise RuntimeError("Agent loop exited without a terminal result")

    @staticmethod
    def _finish_stop_reason(response: ModelResponse) -> str:
        """Map normal provider finish markers while preserving abnormal markers."""
        if response.finish_reason in NORMAL_FINISH_REASONS:
            return COMPLETED_STOP_REASON
        return response.finish_reason or COMPLETED_STOP_REASON

    @staticmethod
    def _append_assistant_message(state: AgentState, response: ModelResponse) -> None:
        state.messages.append(
            Message(
                role="assistant",
                content=response.text,
                reasoning_content=response.reasoning_content,
                tool_calls=response.tool_calls,
            )
        )
