import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from coding_agent import Message, ToolCall
from coding_agent import session_store as session_store_module
from coding_agent.session_store import (
    DEFAULT_MAX_SESSION_BYTES,
    SESSION_SCHEMA_VERSION,
    SessionArchive,
    SessionConflictError,
    SessionPathError,
    SessionSizeError,
    SessionStore,
    SessionStoreError,
    SessionValidationError,
)


def complete_history() -> list[Message]:
    call = ToolCall(id="call_read", name="read_file", arguments={"path": "notes.txt"})
    return [
        Message(role="system", content="rules"),
        Message(role="user", content="read the note"),
        Message(role="assistant", content=None, tool_calls=[call]),
        Message(role="tool", tool_call_id=call.id, content="note contents"),
        Message(role="assistant", content="the note says note contents"),
    ]


def test_session_store_round_trips_versioned_complete_history(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    now = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)

    archive = store.create("chat_1", tmp_path, complete_history(), now=now)

    assert archive.schema_version == SESSION_SCHEMA_VERSION
    assert archive.revision == 0
    assert archive.workspace_root == str(tmp_path.resolve())
    assert store.load("chat_1", workspace_root=tmp_path) == archive
    payload = json.loads(store.path_for("chat_1").read_text(encoding="utf-8"))
    assert set(payload) == {
        "created_at",
        "messages",
        "revision",
        "schema_version",
        "session_id",
        "updated_at",
        "workspace_root",
    }
    assert "api_key" not in payload


def test_session_store_refuses_duplicate_and_stale_updates(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    archive = store.create("chat_1", tmp_path, [], now=datetime.now(UTC))

    with pytest.raises(SessionConflictError):
        store.create("chat_1", tmp_path, [])

    updated = store.save(archive, now=datetime.now(UTC) + timedelta(seconds=1))
    assert updated.revision == 1
    with pytest.raises(SessionConflictError):
        store.save(archive)


@pytest.mark.parametrize(
    "messages",
    [
        [
            Message(role="user", content="unfinished"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id="call_missing", name="read_file", arguments={"path": "x"})],
            ),
        ],
        [
            Message(role="user", content="mismatched"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(id="call_expected", name="read_file", arguments={"path": "x"})
                ],
            ),
            Message(role="tool", tool_call_id="call_other", content="result"),
        ],
    ],
)
def test_session_archive_rejects_incomplete_or_mismatched_history(
    tmp_path: Path, messages: list[Message]
) -> None:
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(SessionValidationError):
        store.create("chat_1", tmp_path, messages)
    assert not store.path_for("chat_1").exists()


def test_session_store_hides_invalid_archive_contents(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    path = store.path_for("bad")
    path.write_text(
        json.dumps(
            {
                "schema_version": SESSION_SCHEMA_VERSION,
                "session_id": "bad",
                "workspace_root": str(tmp_path.resolve()),
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "revision": 0,
                "messages": [
                    {"role": "user", "content": "private secret text"},
                    {"role": "assistant", "content": None, "tool_calls": []},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SessionValidationError) as error:
        store.load("bad")
    assert "private secret text" not in str(error.value)


@pytest.mark.parametrize("session_id", ["", ".", "../escape", "bad/name", "CON", "a" * 65])
def test_session_store_rejects_unsafe_session_ids(tmp_path: Path, session_id: str) -> None:
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(ValueError, match="invalid session id"):
        store.path_for(session_id)


def test_session_store_rejects_workspace_mismatch(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    store.create("chat_1", tmp_path / "workspace-a", [])

    with pytest.raises(SessionConflictError, match="workspace"):
        store.load("chat_1", workspace_root=tmp_path / "workspace-b")


def test_session_store_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    archive = store.create("chat_1", tmp_path, [])
    path = store.path_for(archive.session_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SessionValidationError):
        store.load("chat_1")


def test_session_store_rejects_existing_session_lock(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    lock_path = store.root / ".chat_1.lock"
    lock_path.write_text("other process", encoding="utf-8")

    with pytest.raises(SessionConflictError, match="in use"):
        store.create("chat_1", tmp_path, [])
    assert lock_path.read_text(encoding="utf-8") == "other process"


def test_session_store_revalidates_mutated_archive_before_save(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    archive = store.create("chat_1", tmp_path, [])
    original_bytes = store.path_for("chat_1").read_bytes()
    archive.messages.append(Message(role="user", content="unfinished"))

    with pytest.raises(SessionValidationError):
        store.save(archive)
    assert store.path_for("chat_1").read_bytes() == original_bytes


def test_session_store_enforces_archive_size_budget(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions", max_bytes=256)
    messages = [
        Message(role="user", content="large task"),
        Message(role="assistant", content="x" * 2_000),
    ]

    with pytest.raises(SessionSizeError):
        store.create("large", tmp_path, messages)


def test_session_store_atomic_failure_preserves_previous_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "sessions")
    archive = store.create("chat_1", tmp_path, [], now=datetime.now(UTC))
    original_bytes = store.path_for("chat_1").read_bytes()
    replacement = SessionArchive(
        schema_version=archive.schema_version,
        session_id=archive.session_id,
        workspace_root=archive.workspace_root,
        created_at=archive.created_at,
        updated_at=archive.updated_at,
        revision=archive.revision,
        messages=[
            Message(role="user", content="new"),
            Message(role="assistant", content="new answer"),
        ],
    )

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(session_store_module.os, "replace", fail_replace)
    with pytest.raises(SessionStoreError):
        store.save(replacement)

    assert store.path_for("chat_1").read_bytes() == original_bytes
    assert list(store.root.glob("*.tmp")) == []


def test_session_store_rejects_storage_root_symlink_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(SessionPathError):
        SessionStore(link)


def test_default_session_size_budget_is_32_mib() -> None:
    assert DEFAULT_MAX_SESSION_BYTES == 32 * 1024 * 1024
