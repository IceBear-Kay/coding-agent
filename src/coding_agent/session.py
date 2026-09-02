"""In-memory session coordination for consecutive completed tasks."""

import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from coding_agent.agent import COMPLETED_STOP_REASON, AgentLoop, AgentRunResult
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
        self.state = SessionState(messages=restored_messages)

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
        self._prepare_title(task)
        result = self.loop.run(task, history=self.state.messages)
        if result.stop_reason == COMPLETED_STOP_REASON:
            committed_messages = [
                message.model_copy(deep=True) for message in result.state.messages
            ]
            self.state.messages = committed_messages
            if self.store is not None and self._archive is not None:
                candidate = self._archive.model_copy(
                    update={"messages": committed_messages},
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
]
