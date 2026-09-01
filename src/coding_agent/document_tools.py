"""Bounded, read-only extraction of text from PDF and DOCX files."""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
import threading
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from multiprocessing.connection import Connection
from pathlib import Path, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from coding_agent.tools import ToolOutput, ToolSpec, Workspace, WorkspacePathError

MAX_DOCUMENT_SOURCE_BYTES = 5 * 1024 * 1024
MAX_DOCUMENT_TEXT_CHARS = 32_000
MAX_DOCUMENT_PAGES = 20
MAX_DOCUMENT_PARSE_SECONDS = 10.0
MAX_DOCUMENT_RESULT_BYTES = 256 * 1024
MAX_DOCX_ENTRIES = 10_000
MAX_DOCX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DOCX_ENTRY_BYTES = 16 * 1024 * 1024
DOCUMENT_TRUNCATION_MARKER = "\n...[document text truncated]"


@dataclass(frozen=True, slots=True)
class _DocumentPathFingerprint:
    """Identity and metadata captured for one path during a document read."""

    path: Path
    device: int
    inode: int
    mode: int
    modified_ns: int
    changed_ns: int
    size: int


@dataclass(frozen=True, slots=True)
class _DocumentPathState:
    target: _DocumentPathFingerprint
    parents: tuple[_DocumentPathFingerprint, ...]


class ReadDocumentArguments(BaseModel):
    """Arguments accepted by the read_document tool."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    start_page: StrictInt | None = Field(default=None, ge=1)
    end_page: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_page_range(self) -> ReadDocumentArguments:
        if isinstance(self.start_page, bool) or isinstance(self.end_page, bool):
            raise ValueError("page numbers must be integers")
        start_page = self.start_page or 1
        if self.end_page is not None:
            if self.end_page < start_page:
                raise ValueError("end_page must not be before start_page")
            if self.end_page - start_page + 1 > MAX_DOCUMENT_PAGES:
                raise ValueError(f"a maximum of {MAX_DOCUMENT_PAGES} pages may be read")
        return self


class _DocumentReadError(Exception):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        result = os.lstat(path)
    except OSError:
        return False
    return bool(getattr(result, "st_reparse_tag", 0))


def _resolve_document_path(workspace: Workspace, requested: str) -> Path:
    target = workspace.resolve_path(requested)
    lexical = workspace.root / Path(requested)
    try:
        relative_parts = lexical.relative_to(workspace.root).parts
    except ValueError as exc:  # Defensive: resolve_path already checks this boundary.
        raise WorkspacePathError("Workspace path must be relative") from exc

    current = workspace.root
    for part in relative_parts:
        current /= part
        if current.exists() and _is_reparse_point(current):
            raise WorkspacePathError("Document path crosses a symbolic link or reparse point")
    if _is_reparse_point(target):
        raise WorkspacePathError("Document path is a symbolic link or reparse point")
    return target


def _fingerprint(path: Path, *, expect_directory: bool) -> _DocumentPathFingerprint:
    try:
        path_stat = os.lstat(path)
    except OSError as exc:
        raise WorkspacePathError("Document path changed while it was being inspected") from exc
    if _is_reparse_point(path):
        raise WorkspacePathError("Document path crosses a symbolic link or reparse point")
    if expect_directory and not stat.S_ISDIR(path_stat.st_mode):
        raise WorkspacePathError("Document parent is no longer a directory")
    if not expect_directory and not stat.S_ISREG(path_stat.st_mode):
        raise WorkspacePathError("Document path is not a regular file")
    return _DocumentPathFingerprint(
        path=path,
        device=path_stat.st_dev,
        inode=path_stat.st_ino,
        mode=path_stat.st_mode,
        modified_ns=path_stat.st_mtime_ns,
        changed_ns=path_stat.st_ctime_ns,
        size=path_stat.st_size,
    )


def _capture_path_state(target: Path, workspace_root: Path) -> _DocumentPathState:
    """Capture target and every parent identity up to the workspace root."""
    try:
        target.relative_to(workspace_root)
    except ValueError as exc:
        raise WorkspacePathError("Document path escapes the workspace root") from exc

    target_fingerprint = _fingerprint(target, expect_directory=False)
    parents: list[_DocumentPathFingerprint] = []
    current = target.parent
    while True:
        parents.append(_fingerprint(current, expect_directory=True))
        if current == workspace_root:
            break
        try:
            current = current.parent
            current.relative_to(workspace_root)
        except ValueError as exc:
            raise WorkspacePathError("Document parent escapes the workspace root") from exc
    return _DocumentPathState(target=target_fingerprint, parents=tuple(parents))


def _fingerprint_matches(
    fingerprint: _DocumentPathFingerprint,
    path_stat: os.stat_result,
) -> bool:
    return (
        fingerprint.device == path_stat.st_dev
        and fingerprint.inode == path_stat.st_ino
        and fingerprint.mode == path_stat.st_mode
        and fingerprint.modified_ns == path_stat.st_mtime_ns
        and fingerprint.changed_ns == path_stat.st_ctime_ns
        and fingerprint.size == path_stat.st_size
    )


def _path_state_matches(state: _DocumentPathState) -> bool:
    try:
        if not _fingerprint_matches(state.target, os.lstat(state.target.path)):
            return False
        return all(_fingerprint_matches(parent, os.lstat(parent.path)) for parent in state.parents)
    except OSError:
        return False


def _descriptor_matches(
    fingerprint: _DocumentPathFingerprint,
    descriptor_stat: os.stat_result,
) -> bool:
    return (
        fingerprint.device == descriptor_stat.st_dev
        and fingerprint.inode == descriptor_stat.st_ino
        and fingerprint.mode == descriptor_stat.st_mode
        and fingerprint.size == descriptor_stat.st_size
    )


def _read_source(workspace: Workspace, requested: str) -> tuple[Path, bytes]:
    try:
        target = _resolve_document_path(workspace, requested)
    except WorkspacePathError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise WorkspacePathError("Document path could not be inspected") from exc

    if not target.exists():
        raise _DocumentReadError("not_found", f"Workspace document does not exist: {requested}")
    if not target.is_file():
        raise _DocumentReadError("not_a_file", f"Workspace path is not a file: {requested}")
    state = _capture_path_state(target, workspace.root)
    try:
        with target.open("rb") as source:
            descriptor_stat = os.fstat(source.fileno())
            if not _descriptor_matches(state.target, descriptor_stat) or not _path_state_matches(
                state
            ):
                raise WorkspacePathError("Document path changed before it was read")
            if descriptor_stat.st_size > MAX_DOCUMENT_SOURCE_BYTES:
                raise _DocumentReadError(
                    "too_large",
                    f"Document exceeds the {MAX_DOCUMENT_SOURCE_BYTES}-byte limit.",
                )
            data = source.read(MAX_DOCUMENT_SOURCE_BYTES + 1)
            if not _descriptor_matches(state.target, os.fstat(source.fileno())):
                raise WorkspacePathError("Document file changed while it was being read")
            if not _path_state_matches(state):
                raise WorkspacePathError("Document parent path changed while it was being read")
    except _DocumentReadError:
        raise
    except OSError as exc:
        raise _DocumentReadError("read_error", "Document could not be read") from exc
    if len(data) > MAX_DOCUMENT_SOURCE_BYTES:
        raise _DocumentReadError(
            "too_large",
            f"Document exceeds the {MAX_DOCUMENT_SOURCE_BYTES}-byte limit.",
        )
    return target, data


def _validate_docx_container(data: bytes) -> None:
    if not zipfile.is_zipfile(BytesIO(data)):
        raise _DocumentReadError("invalid_document", "DOCX container is invalid")
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(infos) > MAX_DOCX_ENTRIES:
                raise _DocumentReadError("resource_limit", "DOCX contains too many entries")
            if len(names) != len(set(names)):
                raise _DocumentReadError("invalid_document", "DOCX contains duplicate entries")
            total_size = 0
            for info in infos:
                name = info.filename.replace("\\", "/")
                windows_name = PureWindowsPath(name)
                if (
                    name.startswith("/")
                    or name.startswith("../")
                    or "/../" in name
                    or windows_name.is_absolute()
                    or windows_name.drive
                ):
                    raise _DocumentReadError(
                        "invalid_document", "DOCX contains an unsafe entry path"
                    )
                if info.file_size > MAX_DOCX_ENTRY_BYTES:
                    raise _DocumentReadError("resource_limit", "DOCX entry exceeds the size limit")
                total_size += info.file_size
                if total_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise _DocumentReadError(
                        "resource_limit", "DOCX uncompressed size exceeds the limit"
                    )
            required = {"[Content_Types].xml", "word/document.xml"}
            if not required.issubset(names):
                raise _DocumentReadError("invalid_document", "DOCX document parts are missing")
    except _DocumentReadError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise _DocumentReadError("invalid_document", "DOCX container could not be read") from exc


def _append_bounded_block(
    blocks: list[str],
    block: str,
    current_chars: int,
) -> tuple[int, bool]:
    """Append one text block while keeping the worker result bounded."""
    if not block:
        return current_chars, False
    separator = 1 if blocks else 0
    available = MAX_DOCUMENT_TEXT_CHARS + 1 - current_chars - separator
    if available <= 0:
        return current_chars, True
    if len(block) > available:
        blocks.append(block[:available])
        return current_chars + separator + available, True
    blocks.append(block)
    return current_chars + separator + len(block), False


def _extract_pdf(data: bytes, start_page: int, end_page: int | None) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data), strict=False)
    if reader.is_encrypted:
        raise _DocumentReadError("encrypted", "Encrypted PDF documents are not supported")
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise _DocumentReadError("no_text", "PDF has no pages or extractable text")
    requested_end = end_page if end_page is not None else start_page + MAX_DOCUMENT_PAGES - 1
    if start_page > total_pages:
        raise _DocumentReadError("page_range", "Requested PDF page range is outside the document")
    actual_end = min(requested_end, total_pages)
    blocks: list[str] = []
    text_chars = 0
    text_truncated = False
    selected_text_pages = 0
    processed_end = start_page - 1
    for number in range(start_page, actual_end + 1):
        processed_end = number
        text = reader.pages[number - 1].extract_text() or ""
        if text:
            selected_text_pages += 1
            text_chars, text_truncated = _append_bounded_block(blocks, text, text_chars)
            if text_truncated:
                break
    return {
        "text": "\n".join(blocks),
        "text_truncated": text_truncated,
        "total_pages": total_pages,
        "selected_pages": actual_end - start_page + 1,
        "processed_pages": max(0, processed_end - start_page + 1),
        "selected_text_pages": selected_text_pages,
        "requested_start_page": start_page,
        "requested_end_page": requested_end,
        "processed_start_page": start_page,
        "processed_end_page": processed_end,
        "page_truncated": requested_end < total_pages,
    }


def _extract_docx(data: bytes) -> dict[str, Any]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(BytesIO(data))
    blocks: list[str] = []
    text_chars = 0
    text_truncated = False
    paragraph_count = 0
    table_count = 0
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph_count += 1
            text = Paragraph(child, document).text
            if text:
                text_chars, text_truncated = _append_bounded_block(blocks, text, text_chars)
        elif child.tag.endswith("}tbl"):
            table_count += 1
            table = Table(child, document)
            rows = ["\t".join(cell.text for cell in row.cells) for row in table.rows]
            if rows:
                text_chars, text_truncated = _append_bounded_block(
                    blocks,
                    "\n".join(rows),
                    text_chars,
                )
        if text_truncated:
            break
    return {
        "text": "\n".join(blocks),
        "text_truncated": text_truncated,
        "paragraphs": paragraph_count,
        "tables": table_count,
        "location": "document order",
    }


def _encode_result(result: dict[str, Any]) -> bytes:
    return json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _bounded_result_payload(result: dict[str, Any]) -> bytes:
    """Serialize a parser result without allowing an oversized IPC payload."""
    candidate = dict(result)
    text = candidate.get("text")
    if isinstance(text, str) and len(text) > MAX_DOCUMENT_TEXT_CHARS + 1:
        candidate["text"] = text[: MAX_DOCUMENT_TEXT_CHARS + 1]
        candidate["text_truncated"] = True
    payload = _encode_result(candidate)
    if len(payload) <= MAX_DOCUMENT_RESULT_BYTES:
        return payload

    if isinstance(text, str):
        text_candidate = candidate["text"]
        low = 0
        high = len(text_candidate)
        best = b""
        while low <= high:
            middle = (low + high) // 2
            candidate["text"] = text_candidate[:middle]
            candidate["text_truncated"] = True
            trial = _encode_result(candidate)
            if len(trial) <= MAX_DOCUMENT_RESULT_BYTES:
                best = trial
                low = middle + 1
            else:
                high = middle - 1
        if best:
            return best
    return _encode_result(
        {
            "error_status": "resource_limit",
            "error_message": "Document parser result exceeded the size limit",
        }
    )


def _sanitize_worker_environment() -> None:
    """Keep parser workers independent from model and unrelated credentials."""
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key.casefold() in {"path", "pathext", "systemroot", "windir", "temp", "tmp"}
    }
    os.environ.clear()
    os.environ.update(allowed)


def _parse_worker(
    format_name: str,
    data: bytes,
    start_page: int | None,
    end_page: int | None,
    result_connection: Connection,
) -> None:
    try:
        _sanitize_worker_environment()
        if format_name == "pdf":
            result = _extract_pdf(data, start_page or 1, end_page)
        else:
            result = _extract_docx(data)
    except _DocumentReadError as exc:
        result = {"error_status": exc.status, "error_message": exc.message}
    except Exception:
        result = {"error_status": "parse_error", "error_message": "Document parsing failed"}
    try:
        result_connection.send_bytes(_bounded_result_payload(result))
    except (BrokenPipeError, OSError):
        pass
    finally:
        result_connection.close()


def _terminate_parse_process(process: multiprocessing.Process) -> bool:
    if not process.is_alive():
        return True
    try:
        process.terminate()
    except (OSError, RuntimeError):
        return False
    process.join(0.5)
    if process.is_alive():
        killer = getattr(process, "kill", None)
        if callable(killer):
            try:
                killer()
            except (OSError, RuntimeError):
                return False
            process.join(0.5)
    return not process.is_alive()


def _receive_result_with_deadline(
    connection: Connection,
    deadline: float,
) -> tuple[bytes | None, BaseException | None, bool]:
    """Receive one framed result without allowing a partial frame to block forever."""
    outcome: dict[str, bytes | BaseException] = {}
    completed = threading.Event()

    def receive() -> None:
        try:
            outcome["payload"] = connection.recv_bytes(MAX_DOCUMENT_RESULT_BYTES)
        except BaseException as exc:  # The main thread maps the receive failure.
            outcome["error"] = exc
        finally:
            completed.set()

    receiver = threading.Thread(target=receive, name="document-result-receiver", daemon=True)
    receiver.start()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            connection.close()
            receiver.join(0.5)
            return None, None, True
        if completed.wait(min(0.05, remaining)):
            receiver.join(0.1)
            payload = outcome.get("payload")
            error = outcome.get("error")
            return (
                payload if isinstance(payload, bytes) else None,
                error if isinstance(error, BaseException) else None,
                False,
            )


def _parse_document(
    format_name: str,
    data: bytes,
    start_page: int | None,
    end_page: int | None,
    timeout_seconds: float = MAX_DOCUMENT_PARSE_SECONDS,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_parse_worker,
        args=(format_name, data, start_page, end_page, send_connection),
    )
    started = False
    try:
        process.start()
        started = True
        send_connection.close()
        deadline = time.monotonic() + timeout_seconds
        payload, receive_error, timed_out = _receive_result_with_deadline(
            receive_connection,
            deadline,
        )
        if timed_out:
            terminated = _terminate_parse_process(process)
            if not terminated:
                raise _DocumentReadError(
                    "parse_error", "Document parser could not be terminated safely"
                )
            raise _DocumentReadError("parse_timeout", "Document parsing exceeded the time limit")
        if receive_error is not None:
            if isinstance(receive_error, EOFError):
                raise _DocumentReadError(
                    "parse_error", "Document parser returned no result"
                ) from receive_error
            if isinstance(receive_error, OSError):
                terminated = _terminate_parse_process(process)
                if not terminated:
                    raise _DocumentReadError(
                        "parse_error", "Document parser could not be terminated safely"
                    ) from receive_error
                raise _DocumentReadError(
                    "resource_limit", "Document parser result exceeded the size limit"
                ) from receive_error
            raise _DocumentReadError(
                "parse_error", "Document parser returned invalid data"
            ) from receive_error
        if payload is None:
            raise _DocumentReadError("parse_error", "Document parser returned no result")
        try:
            result = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise _DocumentReadError(
                "parse_error", "Document parser returned invalid data"
            ) from exc
        if not isinstance(result, dict):
            raise _DocumentReadError("parse_error", "Document parser returned invalid data")
        remaining = max(0.0, deadline - time.monotonic())
        process.join(min(1.0, remaining))
        if process.is_alive() and not _terminate_parse_process(process):
            raise _DocumentReadError(
                "parse_error", "Document parser could not be terminated safely"
            )
        if "error_status" in result:
            raise _DocumentReadError(result["error_status"], result["error_message"])
        return result
    except _DocumentReadError:
        raise
    except (OSError, RuntimeError) as exc:
        raise _DocumentReadError("parse_error", "Document parser could not be started") from exc
    finally:
        if started and process.is_alive():
            _terminate_parse_process(process)
        receive_connection.close()
        if not started:
            send_connection.close()


def _truncate_text(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_DOCUMENT_TEXT_CHARS:
        return text, False
    limit = MAX_DOCUMENT_TEXT_CHARS - len(DOCUMENT_TRUNCATION_MARKER)
    return text[:limit] + DOCUMENT_TRUNCATION_MARKER, True


class _ReadDocumentHandler:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def __call__(self, arguments: ReadDocumentArguments) -> ToolOutput:
        try:
            target, data = _read_source(self.workspace, arguments.path)
            format_name = target.suffix.casefold().lstrip(".")
            if format_name not in {"pdf", "docx"}:
                return self._error(
                    "unsupported_format",
                    arguments.path,
                    "Only PDF and DOCX documents are supported.",
                )
            if format_name == "docx" and (
                arguments.start_page is not None or arguments.end_page is not None
            ):
                return self._error(
                    "invalid_arguments",
                    arguments.path,
                    "Page range arguments apply only to PDF documents.",
                )
            if format_name == "docx":
                _validate_docx_container(data)
            parsed = _parse_document(
                format_name,
                data,
                arguments.start_page,
                arguments.end_page,
            )
            parser_text_truncated = bool(parsed.pop("text_truncated", False))
            text, text_truncated = _truncate_text(parsed.pop("text", ""))
            page_truncated = bool(parsed.pop("page_truncated", False))
            truncation_reasons: list[str] = []
            if page_truncated:
                truncation_reasons.append("page_limit")
            if parser_text_truncated or text_truncated:
                truncation_reasons.append("text_limit")
            if not text:
                if format_name == "pdf":
                    message = (
                        "The selected PDF pages contain no extractable text; "
                        "unread pages may contain text."
                    )
                else:
                    message = (
                        "Document has no extractable text; it may be scanned or "
                        "contain no text layer."
                    )
                details = {
                    "path": arguments.path,
                    "format": format_name,
                    "text": "",
                    "truncated": bool(truncation_reasons),
                    "message": message,
                    **parsed,
                }
                if truncation_reasons:
                    details["truncation_reasons"] = truncation_reasons
                return ToolOutput(status="no_text", details=details, is_error=True)
            details: dict[str, Any] = {
                "path": arguments.path,
                "format": format_name,
                "text": text,
                "truncated": bool(truncation_reasons),
                **parsed,
            }
            if truncation_reasons:
                details["truncation_reasons"] = truncation_reasons
            return ToolOutput(status="completed", details=details)
        except WorkspacePathError as exc:
            return self._error("invalid_path", arguments.path, str(exc))
        except _DocumentReadError as exc:
            return self._error(exc.status, arguments.path, exc.message)
        except (OSError, RuntimeError, ValueError):
            return self._error("parse_error", arguments.path, "Document parsing failed")

    @staticmethod
    def _error(status: str, path: str, message: str) -> ToolOutput:
        return ToolOutput(status=status, details={"path": path, "message": message}, is_error=True)


def read_document_tool_spec(workspace: Workspace) -> ToolSpec[ReadDocumentArguments]:
    """Build the read-only PDF/DOCX extraction tool specification."""
    return ToolSpec(
        name="read_document",
        description="Extract bounded text from a PDF or DOCX document in the workspace.",
        parameters=ReadDocumentArguments,
        handler=_ReadDocumentHandler(workspace),
    )


__all__ = [
    "DOCUMENT_TRUNCATION_MARKER",
    "MAX_DOCUMENT_PAGES",
    "MAX_DOCUMENT_PARSE_SECONDS",
    "MAX_DOCUMENT_RESULT_BYTES",
    "MAX_DOCUMENT_SOURCE_BYTES",
    "MAX_DOCUMENT_TEXT_CHARS",
    "MAX_DOCX_ENTRIES",
    "MAX_DOCX_ENTRY_BYTES",
    "MAX_DOCX_UNCOMPRESSED_BYTES",
    "ReadDocumentArguments",
    "read_document_tool_spec",
]
