"""In-memory session coordination for consecutive completed tasks."""

import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from coding_agent.agent import COMPLETED_STOP_REASON, AgentLoop, AgentRunResult
from coding_agent.compaction import (
    COMPACTION_AUTO_THRESHOLD,
    CompactionResult,
    apply_compaction_view,
    compact_history,
    compaction_message,
    compaction_prefix_matches,
)
from coding_agent.models import AgentState, Message, SessionState, Usage
from coding_agent.session_store import (
    MAX_SESSION_TITLE_LENGTH,
    SessionArchive,
    SessionConflictError,
    SessionLease,
    SessionStore,
    SessionStoreError,
)

SESSION_SAVE_ERROR_STOP_REASON = "session_save_error"
SESSION_LOCK_ERROR_STOP_REASON = "session_lock_error"


@dataclass
class SessionTitleStats:
    """Usage isolated from the main Agent Loop task budget and statistics."""

    provider_attempts: int = 0
    known_usage_requests: int = 0
    unknown_usage_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    fallback_used: bool = False


@dataclass
class SessionCompactionStats:
    """Usage isolated from the main task budget for summary requests."""

    provider_attempts: int = 0
    known_usage_requests: int = 0
    unknown_usage_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentSession:
    """Keep one authoritative history while each task gets a fresh AgentState."""

    def __init__(
        self,
        loop: AgentLoop,
        *,
        store: SessionStore | None = None,
        archive: SessionArchive | None = None,
        lease: SessionLease | None = None,
        allow_auto_title: bool = False,
    ) -> None:
        if (store is None) != (archive is None):
            raise ValueError("store and archive must be provided together")
        if archive is None and lease is not None:
            raise ValueError("lease requires a persistent archive")
        if archive is not None and (lease is None or not lease.held):
            raise ValueError("persistent session requires an active lease")
        self.loop = loop
        self.store = store
        self._archive = archive
        self._lease = lease
        self._allow_auto_title = allow_auto_title
        self.title_stats = SessionTitleStats()
        self.compaction_stats = SessionCompactionStats()
        self.last_compaction_result: CompactionResult | None = None
        if archive is not None:
            if os.path.normcase(archive.workspace_root) != os.path.normcase(
                str(loop.workspace.root)
            ):
                raise ValueError("session workspace does not match loop workspace")
            with suppress(ValueError):
                loop.workspace.protect_path(store.root)  # type: ignore[union-attr]
        restored_messages = []
        if archive is not None:
            restored_messages = [
                message.model_copy(deep=True)
                for message in archive.messages
                if self.loop.system_prompt is None or message.role != "system"
            ]
        self.state = SessionState(
            messages=restored_messages,
            compaction=archive.compaction if archive else None,
        )

    @classmethod
    def create(cls, loop: AgentLoop, store: SessionStore, session_id: str) -> "AgentSession":
        """Create an empty persisted session without calling the provider."""
        archive = store.create(session_id, loop.workspace.root)
        lease: SessionLease | None = None
        try:
            lease = store.acquire(session_id)
            return cls(
                loop,
                store=store,
                archive=archive,
                lease=lease,
                allow_auto_title=True,
            )
        except BaseException:
            if lease is not None:
                with suppress(Exception):
                    lease.release()
            raise

    @classmethod
    def resume(cls, loop: AgentLoop, store: SessionStore, session_id: str) -> "AgentSession":
        """Load completed history without replaying model or tool calls."""
        lease = store.acquire(session_id)
        try:
            archive = store.load(session_id, workspace_root=loop.workspace.root)
            return cls(loop, store=store, archive=archive, lease=lease)
        except BaseException:
            with suppress(Exception):
                lease.release()
            raise

    @property
    def session_id(self) -> str | None:
        """Return the persistent session ID, or ``None`` for memory-only sessions."""
        return self._archive.session_id if self._archive is not None else None

    @property
    def archive_path(self) -> Path | None:
        """Return the persistent archive path when persistence is enabled."""
        return (
            self.store.path_for(self._archive.session_id) if self.store and self._archive else None
        )

    @property
    def title(self) -> str | None:
        """Return the persisted user-visible title, if one exists."""
        return self._archive.title if self._archive is not None else None

    @property
    def messages(self) -> list[Message]:
        """Expose the committed history for inspection without changing ownership."""
        return self.state.messages

    def run(self, task: str) -> AgentRunResult:
        """Run a task and commit its complete history only after normal completion."""
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        if self._archive is not None and (self._lease is None or not self._lease.held):
            state = AgentState(
                workspace_root=Path(self.loop.workspace.root),
                max_steps=self.loop.max_steps,
                stop_reason=SESSION_LOCK_ERROR_STOP_REASON,
            )
            state.stats.stop_reason = SESSION_LOCK_ERROR_STOP_REASON
            return AgentRunResult(
                answer=None,
                state=state,
                error=SessionConflictError("session lease is no longer held"),
            )
        self._maybe_auto_compact(task)
        self._prepare_title(task)
        context_prefix: list[Message] | None = None
        compacted_task_count = 0
        if (
            self.state.compaction is not None
            and self.loop.context_policy == "compact"
            and compaction_prefix_matches(self.state.messages, self.state.compaction)
        ):
            context_prefix = [compaction_message(self.state.compaction)]
            compacted_task_count = self.state.compaction.covered_task_count
        result = self.loop.run(
            task,
            history=self.state.messages,
            context_prefix=context_prefix,
            compacted_task_count=compacted_task_count,
        )
        if result.stop_reason == COMPLETED_STOP_REASON:
            committed_messages = [
                message.model_copy(deep=True) for message in result.state.messages
            ]
            self.state.messages = committed_messages
            if self.store is not None and self._archive is not None:
                candidate = self._archive.model_copy(
                    update={"messages": committed_messages, "compaction": self.state.compaction},
                    deep=True,
                )
                try:
                    self._archive = self.store.save(candidate, lease=self._lease)
                except SessionStoreError as exc:
                    result.state.stop_reason = SESSION_SAVE_ERROR_STOP_REASON
                    result.state.stats.stop_reason = SESSION_SAVE_ERROR_STOP_REASON
                    return AgentRunResult(
                        answer=result.answer,
                        state=result.state,
                        error=exc,
                    )
        return result

    def rename(self, title: str) -> None:
        """Persist a manual title without changing conversation history."""
        if self._archive is None or self.store is None or self._lease is None:
            raise SessionStoreError("session is not persistent")
        candidate = self._archive.model_copy(
            update={"title": title, "title_generation_attempted": True},
            deep=True,
        )
        self._archive = self.store.save(candidate, lease=self._lease)

    def _prepare_title(self, task: str) -> None:
        """Attempt automatic naming once, while keeping the main task independent."""
        if self._archive is None or self.store is None or self._lease is None:
            return
        if not self._allow_auto_title:
            return
        if self._archive.title_generation_attempted or self._archive.title is not None:
            return

        fallback = " ".join(task.strip().split())[:MAX_SESSION_TITLE_LENGTH] or "新会话"
        title = fallback
        generator = getattr(self.loop.provider, "generate_title", None)
        if callable(generator):
            self.title_stats.provider_attempts = 1
            try:
                response = generator(task)
                self._record_title_usage(response.usage)
                candidate_archive = SessionArchive.model_validate(
                    {
                        **self._archive.model_dump(mode="python"),
                        "title": response.text,
                        "title_generation_attempted": True,
                    }
                )
                title = candidate_archive.title or fallback
            except KeyboardInterrupt:
                self.title_stats.unknown_usage_requests = 1
                raise
            except Exception:
                if self.title_stats.known_usage_requests == 0:
                    self.title_stats.unknown_usage_requests = 1
                title = fallback
                self.title_stats.fallback_used = True

        candidate_archive = self._archive.model_copy(
            update={"title": title, "title_generation_attempted": True},
            deep=True,
        )
        try:
            self._archive = self.store.save(candidate_archive, lease=self._lease)
        except SessionStoreError:
            # The primary task remains authoritative; a later completed-task save
            # can persist the in-memory title without replaying the title request.
            self._archive = candidate_archive

    def _record_title_usage(self, usage: Usage | None) -> None:
        if usage is None:
            self.title_stats.unknown_usage_requests = 1
            return
        self.title_stats.known_usage_requests = 1
        self.title_stats.input_tokens = usage.input_tokens
        self.title_stats.output_tokens = usage.output_tokens
        self.title_stats.total_tokens = usage.total_tokens

    def clear(self) -> None:
        """Discard committed messages while retaining the loop configuration."""
        self.state.messages.clear()
        self.state.compaction = None

    def compact(self) -> CompactionResult:
        """Create one summary for eligible completed tasks without adding a task."""
        if self.loop.context_policy != "compact":
            result = CompactionResult(False, reason="compact_policy_required")
            self.last_compaction_result = result
            return result
        if self._archive is not None and (self._lease is None or not self._lease.held):
            result = CompactionResult(False, reason="session_lock_error")
            self.last_compaction_result = result
            return result
        result = compact_history(
            self.loop.provider,
            self.state.messages,
            previous=self.state.compaction,
            max_context_bytes=self.loop.context_budget.max_bytes,
            max_context_tokens=self.loop.effective_context_tokens,
            tool_schemas=self.loop.registry.schemas(),
        )
        self._record_compaction_usage(result)
        if not result.success or result.record is None:
            self.last_compaction_result = result
            return result
        previous = self.state.compaction
        if self._archive is not None and self.store is not None:
            candidate = self._archive.model_copy(update={"compaction": result.record}, deep=True)
            try:
                self._archive = self.store.save(candidate, lease=self._lease)
            except SessionStoreError:
                result = CompactionResult(
                    False,
                    reason="summary_save_failed",
                    covered_task_count=previous.covered_task_count if previous else 0,
                    previous_task_count=previous.covered_task_count if previous else 0,
                    before_bytes=result.before_bytes,
                    after_bytes=result.after_bytes,
                )
                self.last_compaction_result = result
                return result
        self.state.compaction = result.record
        self.last_compaction_result = result
        return result

    def _maybe_auto_compact(self, task: str) -> CompactionResult | None:
        """Attempt at most one summary before the first main request of a task."""
        if self.loop.context_policy != "compact" or not self.state.messages:
            return None
        system_messages = (
            [Message(role="system", content=self.loop.system_prompt)]
            if self.loop.system_prompt
            and not any(message.role == "system" for message in self.state.messages)
            else []
        )
        history_view = apply_compaction_view(self.state.messages, self.state.compaction)
        candidate = [*system_messages, *history_view, Message(role="user", content=task)]
        budget = self.loop.context_budget.check(candidate, self.loop.registry.schemas())
        threshold_bytes = budget.used_bytes >= int(budget.max_bytes * COMPACTION_AUTO_THRESHOLD)
        threshold_tokens = (
            budget.used_tokens is not None
            and budget.max_tokens is not None
            and budget.used_tokens >= int(budget.max_tokens * COMPACTION_AUTO_THRESHOLD)
        )
        if not (threshold_bytes or threshold_tokens):
            return None
        return self.compact()

    def _record_compaction_usage(self, result: CompactionResult) -> None:
        if result.reason in {
            "compact_policy_required",
            "history_invalid",
            "nothing_to_compact",
            "session_lock_error",
            "summary_input_limit",
        }:
            return
        if result.input_tokens is None or result.output_tokens is None:
            self.compaction_stats.unknown_usage_requests += 1
            self.compaction_stats.provider_attempts += 1
            return
        self.compaction_stats.provider_attempts += 1
        self.compaction_stats.known_usage_requests += 1
        self.compaction_stats.input_tokens += result.input_tokens
        self.compaction_stats.output_tokens += result.output_tokens
        self.compaction_stats.total_tokens += result.input_tokens + result.output_tokens

    def close(self) -> None:
        """Release the persistent session lease, preserving the archive on disk."""
        if self._lease is not None:
            lease = self._lease
            lease.release()
            self._lease = None

    def __enter__(self) -> "AgentSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "AgentSession",
    "SESSION_LOCK_ERROR_STOP_REASON",
    "SESSION_SAVE_ERROR_STOP_REASON",
    "SessionTitleStats",
    "SessionCompactionStats",
]
