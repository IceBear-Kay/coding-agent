"""Bounded, read-only extraction of text from PDF and DOCX files."""

from __future__ import annotations

import multiprocessing
import os
import queue
import time
import zipfile
from io import BytesIO
from multiprocessing.queues import Queue
from pathlib import Path, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from coding_agent.tools import ToolOutput, ToolSpec, Workspace, WorkspacePathError

MAX_DOCUMENT_SOURCE_BYTES = 5 * 1024 * 1024
MAX_DOCUMENT_TEXT_CHARS = 32_000
MAX_DOCUMENT_PAGES = 20
MAX_DOCUMENT_PARSE_SECONDS = 10.0
MAX_DOCX_ENTRIES = 10_000
MAX_DOCX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DOCX_ENTRY_BYTES = 16 * 1024 * 1024
DOCUMENT_TRUNCATION_MARKER = "\n...[document text truncated]"


class ReadDocumentArguments(BaseModel):
    """Arguments accepted by the read_document tool."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    start_page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)

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
    try:
        if target.stat().st_size > MAX_DOCUMENT_SOURCE_BYTES:
            raise _DocumentReadError(
                "too_large",
                f"Document exceeds the {MAX_DOCUMENT_SOURCE_BYTES}-byte limit.",
            )
        with target.open("rb") as source:
            data = source.read(MAX_DOCUMENT_SOURCE_BYTES + 1)
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
    for number in range(start_page, actual_end + 1):
        text = reader.pages[number - 1].extract_text() or ""
        if text:
            blocks.append(text)
    return {
        "text": "\n".join(blocks),
        "total_pages": total_pages,
        "requested_start_page": start_page,
        "requested_end_page": requested_end,
        "processed_start_page": start_page,
        "processed_end_page": actual_end,
        "page_truncated": requested_end < total_pages,
    }


def _extract_docx(data: bytes) -> dict[str, Any]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(BytesIO(data))
    blocks: list[str] = []
    paragraph_count = 0
    table_count = 0
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph_count += 1
            text = Paragraph(child, document).text
            if text:
                blocks.append(text)
        elif child.tag.endswith("}tbl"):
            table_count += 1
            table = Table(child, document)
            rows = ["\t".join(cell.text for cell in row.cells) for row in table.rows]
            if rows:
                blocks.append("\n".join(rows))
    return {
        "text": "\n".join(blocks),
        "paragraphs": paragraph_count,
        "tables": table_count,
        "location": "document order",
    }


def _parse_worker(
    format_name: str,
    data: bytes,
    start_page: int | None,
    end_page: int | None,
    result_queue: Queue,
) -> None:
    try:
        if format_name == "pdf":
            result = _extract_pdf(data, start_page or 1, end_page)
        else:
            result = _extract_docx(data)
    except _DocumentReadError as exc:
        result_queue.put({"error_status": exc.status, "error_message": exc.message})
    except Exception:
        result_queue.put(
            {"error_status": "parse_error", "error_message": "Document parsing failed"}
        )
    else:
        result_queue.put(result)


def _parse_document(
    format_name: str,
    data: bytes,
    start_page: int | None,
    end_page: int | None,
    timeout_seconds: float = MAX_DOCUMENT_PARSE_SECONDS,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    result_queue: Queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_parse_worker,
        args=(format_name, data, start_page, end_page, result_queue),
    )
    started = False
    try:
        process.start()
        started = True
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                process.join(1.0)
                raise _DocumentReadError(
                    "parse_timeout", "Document parsing exceeded the time limit"
                )
            try:
                result = result_queue.get(timeout=min(0.1, remaining))
                break
            except queue.Empty:
                if process.is_alive():
                    continue
                try:
                    result = result_queue.get(timeout=min(0.2, remaining))
                    break
                except queue.Empty as empty_exc:
                    raise _DocumentReadError(
                        "parse_error", "Document parser returned no result"
                    ) from empty_exc
        remaining = max(0.0, deadline - time.monotonic())
        process.join(min(1.0, remaining))
        if process.is_alive():
            process.terminate()
            process.join(1.0)
        if "error_status" in result:
            raise _DocumentReadError(result["error_status"], result["error_message"])
        return result
    except _DocumentReadError:
        raise
    except (OSError, RuntimeError) as exc:
        raise _DocumentReadError("parse_error", "Document parser could not be started") from exc
    finally:
        if started and process.is_alive():
            process.terminate()
            process.join(1.0)
        result_queue.close()
        result_queue.join_thread()


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
            text, text_truncated = _truncate_text(parsed.pop("text", ""))
            if not text:
                details = {
                    "path": arguments.path,
                    "format": format_name,
                    "text": "",
                    "truncated": False,
                    "message": (
                        "Document has no extractable text; it may be scanned or "
                        "contain no text layer."
                    ),
                    **parsed,
                }
                return ToolOutput(status="no_text", details=details, is_error=True)
            reasons: list[str] = []
            if parsed.pop("page_truncated", False):
                reasons.append("page_limit")
            if text_truncated:
                reasons.append("text_limit")
            details: dict[str, Any] = {
                "path": arguments.path,
                "format": format_name,
                "text": text,
                "truncated": bool(reasons),
                **parsed,
            }
            if reasons:
                details["truncation_reasons"] = reasons
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
    "MAX_DOCUMENT_SOURCE_BYTES",
    "MAX_DOCUMENT_TEXT_CHARS",
    "MAX_DOCX_ENTRIES",
    "MAX_DOCX_ENTRY_BYTES",
    "MAX_DOCX_UNCOMPRESSED_BYTES",
    "ReadDocumentArguments",
    "read_document_tool_spec",
]
