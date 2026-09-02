"""Versioned, bounded persistence for completed chat session history."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from coding_agent.context import ContextHistoryError, validate_completed_history
from coding_agent.models import CompactionRecord, Message

SESSION_SCHEMA_VERSION = 1
DEFAULT_MAX_SESSION_BYTES = 32 * 1024 * 1024
MAX_SESSION_ID_LENGTH = 64
MAX_SESSION_TITLE_LENGTH = 60
_SESSION_ID_PATTERN = re.compile(rf"^[A-Za-z0-9][A-Za-z0-9_-]{{0,{MAX_SESSION_ID_LENGTH - 1}}}$")
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class SessionStoreError(Exception):
    """Base class for safe session persistence failures."""


class SessionConflictError(SessionStoreError):
    """Raised when a session already exists or has changed since it was loaded."""


class SessionNotFoundError(SessionStoreError):
    """Raised when a requested session archive does not exist."""


class SessionPathError(SessionStoreError):
    """Raised when a session path is unsafe or redirects outside its store."""


class SessionSizeError(SessionStoreError):
    """Raised when an archive exceeds the configured byte budget."""


class SessionValidationError(SessionStoreError):
    """Raised when an archive cannot be trusted as a completed session."""


class SessionLease:
    """An exclusive lock held for the lifetime of one persistent session."""

    def __init__(self, store: SessionStore, session_id: str) -> None:
        self.store = store
        self.session_id = _validate_session_id(session_id)
        self.path = store.root / f".{self.session_id}.lock"
        self._descriptor: int | None = None
        self._identity: tuple[int, int] | None = None

    @property
    def held(self) -> bool:
        descriptor = self._descriptor
        identity = self._identity
        if descriptor is None or identity is None:
            return False
        return self._path_matches(descriptor, identity)

    def _path_matches(self, descriptor: int, identity: tuple[int, int]) -> bool:
        """Check that the lock path still names this lease's opened file."""
        try:
            current = self.path.lstat()
            descriptor_stat = os.fstat(descriptor)
        except OSError:
            return False
        return (
            (current.st_dev, current.st_ino)
            == identity
            == (
                descriptor_stat.st_dev,
                descriptor_stat.st_ino,
            )
        )

    def acquire(self) -> SessionLease:
        if self.held:
            return self
        self.store._check_path(self.path)
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise SessionConflictError("session is already in use") from exc
        except OSError as exc:
            raise SessionStoreError("session lock could not be created") from exc

        self._descriptor = descriptor
        try:
            descriptor_stat = os.fstat(descriptor)
            self._identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.fsync(descriptor)
        except BaseException as exc:
            cleanup_error: OSError | None = None
            owns_path = self._identity is not None and self._path_matches(
                descriptor, self._identity
            )
            try:
                os.close(descriptor)
            except OSError as close_exc:
                cleanup_error = close_exc
            self._descriptor = None
            self._identity = None
            if owns_path:
                try:
                    self.path.unlink()
                except OSError as unlink_exc:
                    cleanup_error = unlink_exc
            if cleanup_error is not None:
                raise SessionStoreError(
                    "session lock initialization failed and cleanup failed"
                ) from exc
            if isinstance(exc, KeyboardInterrupt):
                raise
            if not isinstance(exc, OSError):
                raise
            raise SessionStoreError("session lock initialization failed") from exc
        return self

    def release(self) -> None:
        descriptor = self._descriptor
        identity = self._identity
        if descriptor is None or identity is None:
            return
        close_error: OSError | None = None
        ownership_error: SessionStoreError | None = None
        owns_path = self._path_matches(descriptor, identity)
        if not owns_path:
            ownership_error = SessionStoreError("session lock ownership changed before release")
        try:
            os.close(descriptor)
        except OSError as exc:
            close_error = exc
        if owns_path:
            try:
                self.path.unlink()
            except OSError as exc:
                if close_error is None:
                    close_error = exc
        self._descriptor = None
        self._identity = None
        if ownership_error is not None:
            raise ownership_error
        if close_error is not None:
            raise SessionStoreError("session lock could not be released") from close_error

    def __enter__(self) -> SessionLease:
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def _validate_session_id(value: str) -> str:
    if not isinstance(value, str) or not _SESSION_ID_PATTERN.fullmatch(value):
        raise ValueError("invalid session id")
    if value.casefold() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("invalid session id")
    return value


def _normalize_workspace_root(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("workspace root is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("workspace root must be absolute")
    return str(path.resolve(strict=False))


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


def _normalize_title(value: str | None) -> str | None:
    """Validate and safely bound user-visible session titles."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("title must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("title must not be empty")
    if any(character.isspace() and character not in {" "} for character in normalized):
        raise ValueError("title must be a single line")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in normalized):
        raise ValueError("title contains control characters")
    return normalized[:MAX_SESSION_TITLE_LENGTH]


class SessionArchive(BaseModel):
    """The only data written to a persisted session archive."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = SESSION_SCHEMA_VERSION
    session_id: str
    workspace_root: str
    created_at: datetime
    updated_at: datetime
    revision: int = Field(default=0, ge=0)
    title: str | None = None
    title_generation_attempted: bool = False
    messages: list[Message] = Field(default_factory=list)
    compaction: CompactionRecord | None = None

    @field_validator("session_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_session_id(value)

    @field_validator("workspace_root")
    @classmethod
    def validate_workspace_root(cls, value: str) -> str:
        return _normalize_workspace_root(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        return _normalize_title(value)

    def model_post_init(self, __context: Any) -> None:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        try:
            validate_completed_history(self.messages)
        except ContextHistoryError as exc:
            raise ValueError("session history is invalid") from exc

    @classmethod
    def new(
        cls,
        session_id: str,
        workspace_root: Path | str,
        messages: Sequence[Message] = (),
        *,
        now: datetime | None = None,
        title: str | None = None,
        title_generation_attempted: bool = False,
        compaction: CompactionRecord | None = None,
    ) -> Self:
        timestamp = _normalize_timestamp(now or datetime.now(UTC))
        try:
            return cls(
                session_id=session_id,
                workspace_root=str(workspace_root),
                created_at=timestamp,
                updated_at=timestamp,
                title=title,
                title_generation_attempted=title_generation_attempted,
                messages=[message.model_copy(deep=True) for message in messages],
                compaction=compaction,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise SessionValidationError("session archive is invalid") from exc

    def to_bytes(self) -> bytes:
        """Serialize using a stable compact UTF-8 JSON representation."""
        try:
            payload = self.model_dump(mode="json", exclude_none=True)
            # Keep archives written before title metadata byte-for-byte compatible.
            if payload.get("title_generation_attempted") is False:
                payload.pop("title_generation_attempted")
            for message in payload["messages"]:
                message.setdefault("content", None)
            return json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise SessionValidationError("session archive serialization failed") from exc

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Parse and validate an archive without exposing its contents in errors."""
        try:
            payload = json.loads(data.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("archive root must be an object")
            try:
                return cls.model_validate(payload)
            except ValidationError as exc:
                # A malformed derived summary must not make an otherwise valid
                # complete history unusable. Drop only that optional field.
                errors = exc.errors()
                compaction_errors_only = bool(errors) and all(
                    (location := error.get("loc", ())) and location[0] == "compaction"
                    for error in errors
                )
                if "compaction" not in payload or not compaction_errors_only:
                    raise
                payload_without_compaction = dict(payload)
                payload_without_compaction.pop("compaction", None)
                return cls.model_validate(payload_without_compaction)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            raise SessionValidationError("session archive is invalid") from exc


class SessionStore:
    """Read and atomically write bounded session archives below one directory."""

    def __init__(
        self,
        root: Path | str,
        *,
        max_bytes: int = DEFAULT_MAX_SESSION_BYTES,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.root = Path(os.path.abspath(os.fspath(Path(root).expanduser())))
        self.max_bytes = max_bytes
        self._ensure_root()

    def path_for(self, session_id: str) -> Path:
        """Return the archive path for a validated ID."""
        _validate_session_id(session_id)
        path = self.root / f"{session_id}.json"
        self._check_path(path)
        return path

    def acquire(self, session_id: str) -> SessionLease:
        """Acquire an exclusive lease for the lifetime of a persistent session."""
        return SessionLease(self, session_id).acquire()

    def create(
        self,
        session_id: str,
        workspace_root: Path | str,
        messages: Sequence[Message] = (),
        *,
        now: datetime | None = None,
    ) -> SessionArchive:
        """Create a new archive and refuse to replace an existing session."""
        archive = SessionArchive.new(session_id, workspace_root, messages, now=now)
        path = self.path_for(archive.session_id)
        with self._session_lock(archive.session_id):
            if path.exists():
                raise SessionConflictError("session already exists")
            self._write_archive(path, archive, replace=False)
        return archive

    def load(
        self,
        session_id: str,
        *,
        workspace_root: Path | str | None = None,
    ) -> SessionArchive:
        """Load one archive with a bounded read and optional workspace check."""
        path = self.path_for(session_id)
        if not path.exists():
            raise SessionNotFoundError("session not found")
        self._check_path(path)
        try:
            data = self._read_bounded(path)
        except SessionStoreError:
            raise
        except OSError as exc:
            raise SessionStoreError("session archive could not be read") from exc
        archive = SessionArchive.from_bytes(data)
        if archive.session_id != session_id:
            raise SessionValidationError("session archive identity mismatch")
        if workspace_root is not None:
            expected = _normalize_workspace_root(str(workspace_root))
            if os.path.normcase(archive.workspace_root) != os.path.normcase(expected):
                raise SessionConflictError("session workspace does not match")
        return archive

    def list_sessions(
        self,
        *,
        workspace_root: Path | str | None = None,
    ) -> tuple[list[SessionArchive], list[str]]:
        """Return readable archives for a workspace and safe skip notices.

        Invalid, oversized, or redirected files are never removed; callers receive
        opaque notices suitable for displaying without exposing archive contents.
        """
        expected = (
            _normalize_workspace_root(str(workspace_root)) if workspace_root is not None else None
        )
        archives: list[SessionArchive] = []
        skipped: list[str] = []
        try:
            candidates = sorted(self.root.glob("*.json"), key=lambda path: path.name.casefold())
        except OSError as exc:
            raise SessionStoreError("session directory could not be listed") from exc
        for path in candidates:
            session_id = path.stem
            try:
                _validate_session_id(session_id)
                archive = self.load(session_id, workspace_root=expected)
            except (SessionStoreError, ValueError):
                skipped.append(f"{path.name}: skipped")
                continue
            archives.append(archive)
        archives.sort(key=lambda archive: archive.updated_at, reverse=True)
        return archives, skipped

    def list(
        self,
        *,
        workspace_root: Path | str | None = None,
    ) -> list[SessionArchive]:
        """List readable sessions while ignoring unsafe or invalid archives."""
        archives, _ = self.list_sessions(workspace_root=workspace_root)
        return archives

    def save(
        self,
        archive: SessionArchive,
        *,
        expected_revision: int | None = None,
        now: datetime | None = None,
        lease: SessionLease | None = None,
    ) -> SessionArchive:
        """Atomically update an existing archive after checking its revision."""
        if not isinstance(archive, SessionArchive):
            raise TypeError("archive must be a SessionArchive")
        path = self.path_for(archive.session_id)
        if lease is not None and (
            lease.store is not self or lease.session_id != archive.session_id or not lease.held
        ):
            raise SessionConflictError("session lease is not held")

        def save_locked() -> SessionArchive:
            current = self.load(archive.session_id)
            expected = archive.revision if expected_revision is None else expected_revision
            if expected != current.revision:
                raise SessionConflictError("session revision conflict")
            if (
                os.path.normcase(archive.workspace_root) != os.path.normcase(current.workspace_root)
                or archive.created_at != current.created_at
            ):
                raise SessionConflictError("session archive identity conflict")
            timestamp = _normalize_timestamp(now or datetime.now(UTC))
            if timestamp < current.updated_at:
                timestamp = current.updated_at
            try:
                updated = SessionArchive.model_validate(
                    {
                        **archive.model_dump(mode="python"),
                        "revision": current.revision + 1,
                        "updated_at": timestamp,
                    }
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise SessionValidationError("session archive is invalid") from exc
            self._write_archive(path, updated, replace=True)
            return updated

        if lease is not None:
            return save_locked()
        with self._session_lock(archive.session_id):
            return save_locked()

    @contextmanager
    def _session_lock(self, session_id: str):
        lease = self.acquire(session_id)
        try:
            yield
        finally:
            lease.release()

    def _ensure_root(self) -> None:
        self._check_existing_components(self.root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionPathError("session directory is unavailable") from exc
        self._check_path(self.root)

    def _check_path(self, path: Path) -> None:
        try:
            self._check_existing_components(path)
        except OSError as exc:
            raise SessionPathError("session path cannot be inspected") from exc
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise SessionPathError("session path escapes store") from exc

    @staticmethod
    def _check_existing_components(path: Path) -> None:
        current = Path(path.anchor) if path.anchor else Path()
        for part in path.parts[1:] if path.anchor else path.parts:
            current /= part
            if not os.path.lexists(current):
                continue
            if current.is_symlink() or _is_reparse_point(current):
                raise SessionPathError("session path cannot contain links")

    def _read_bounded(self, path: Path) -> bytes:
        try:
            with path.open("rb") as stream:
                data = stream.read(self.max_bytes + 1)
        except OSError as exc:
            raise SessionStoreError("session archive could not be read") from exc
        if len(data) > self.max_bytes:
            raise SessionSizeError("session archive exceeds byte limit")
        return data

    def _write_archive(self, path: Path, archive: SessionArchive, *, replace: bool) -> None:
        data = archive.to_bytes()
        if len(data) > self.max_bytes:
            raise SessionSizeError("session archive exceeds byte limit")
        self._check_path(path)
        if not replace and path.exists():
            raise SessionConflictError("session already exists")

        temporary_path: Path | None = None
        save_error: SessionStoreError | None = None
        save_cause: BaseException | None = None
        cleanup_error: OSError | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.root,
                prefix=f".{archive.session_id}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if not replace and path.exists():
                raise SessionConflictError("session already exists")
            os.replace(temporary_path, path)
            temporary_path = None
        except SessionStoreError as exc:
            save_error = exc
        except OSError as exc:
            save_error = SessionStoreError("session archive could not be saved")
            save_cause = exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    cleanup_error = exc
        if save_error is not None and cleanup_error is not None:
            raise SessionStoreError(
                "session archive could not be saved; temporary cleanup failed"
            ) from save_error
        if save_error is not None:
            if save_cause is not None:
                raise save_error from save_cause
            raise save_error
        if cleanup_error is not None:
            raise SessionStoreError("session archive temporary cleanup failed") from cleanup_error


def _is_reparse_point(path: Path) -> bool:
    """Detect Windows junctions/reparse points without affecting other platforms."""
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & 0x400)


__all__ = [
    "DEFAULT_MAX_SESSION_BYTES",
    "MAX_SESSION_ID_LENGTH",
    "MAX_SESSION_TITLE_LENGTH",
    "SESSION_SCHEMA_VERSION",
    "SessionArchive",
    "SessionConflictError",
    "SessionNotFoundError",
    "SessionPathError",
    "SessionSizeError",
    "SessionStore",
    "SessionStoreError",
    "SessionLease",
    "SessionValidationError",
]
