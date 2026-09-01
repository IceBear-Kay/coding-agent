import hashlib
import io
import json
from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from coding_agent.document_tools import (
    MAX_DOCUMENT_PAGES,
    MAX_DOCUMENT_SOURCE_BYTES,
    MAX_DOCUMENT_TEXT_CHARS,
    ReadDocumentArguments,
    read_document_tool_spec,
)
from coding_agent.models import ToolCall, ToolResult
from coding_agent.tools import ToolDispatcher, ToolRegistry, Workspace


def _pdf_bytes(*texts: str) -> bytes:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        stream = DecodedStreamObject()
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode())
        page[NameObject("/Contents")] = stream
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


def _dispatch(workspace: Workspace, call: ToolCall) -> ToolResult:
    registry = ToolRegistry([read_document_tool_spec(workspace)])
    return ToolDispatcher(registry).dispatch(call)


def test_read_document_extracts_pdf_text_and_page_range(tmp_path: Path) -> None:
    target = tmp_path / "notes.pdf"
    target.write_bytes(_pdf_bytes("first page", "second page"))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    result = _dispatch(
        Workspace(tmp_path),
        ToolCall(
            id="call_pdf",
            name="read_document",
            arguments={"path": "notes.pdf", "start_page": 2, "end_page": 2},
        ),
    )

    payload = json.loads(result.content)
    assert result.is_error is False
    assert payload["status"] == "completed"
    assert payload["format"] == "pdf"
    assert "second page" in payload["text"]
    assert "first page" not in payload["text"]
    assert payload["processed_start_page"] == 2
    assert payload["processed_end_page"] == 2
    assert payload["total_pages"] == 2
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest


def test_read_document_extracts_docx_paragraphs_and_tables(tmp_path: Path) -> None:
    target = tmp_path / "notes.docx"
    document = Document()
    document.add_paragraph("Before table")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    table.cell(1, 0).text = "C"
    table.cell(1, 1).text = "D"
    document.add_paragraph("After table")
    document.save(target)

    result = _dispatch(
        Workspace(tmp_path),
        ToolCall(id="call_docx", name="read_document", arguments={"path": "notes.docx"}),
    )

    payload = json.loads(result.content)
    assert result.is_error is False
    assert payload["status"] == "completed"
    assert payload["format"] == "docx"
    assert payload["text"] == "Before table\nA\tB\nC\tD\nAfter table"
    assert payload["paragraphs"] == 2
    assert payload["tables"] == 1


@pytest.mark.parametrize(
    ("filename", "content", "status"),
    [
        ("notes.txt", b"plain", "unsupported_format"),
        ("notes.doc", b"old", "unsupported_format"),
        ("notes.docm", b"macro", "unsupported_format"),
        ("broken.pdf", b"not a pdf", "parse_error"),
        ("broken.docx", b"not a zip", "invalid_document"),
    ],
)
def test_read_document_reports_unsupported_and_damaged_documents(
    tmp_path: Path,
    filename: str,
    content: bytes,
    status: str,
) -> None:
    (tmp_path / filename).write_bytes(content)
    result = _dispatch(
        Workspace(tmp_path),
        ToolCall(id="call_document", name="read_document", arguments={"path": filename}),
    )

    payload = json.loads(result.content)
    assert result.is_error is True
    assert payload["status"] == status


def test_read_document_reports_empty_and_encrypted_pdf(tmp_path: Path) -> None:
    (tmp_path / "empty.pdf").write_bytes(_pdf_bytes(""))
    empty = _dispatch(
        Workspace(tmp_path),
        ToolCall(id="call_empty", name="read_document", arguments={"path": "empty.pdf"}),
    )
    assert json.loads(empty.content)["status"] == "no_text"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    encrypted = io.BytesIO()
    writer.write(encrypted)
    (tmp_path / "encrypted.pdf").write_bytes(encrypted.getvalue())
    locked = _dispatch(
        Workspace(tmp_path),
        ToolCall(id="call_locked", name="read_document", arguments={"path": "encrypted.pdf"}),
    )
    assert json.loads(locked.content)["status"] == "encrypted"


def test_read_document_truncates_text_with_explicit_marker(tmp_path: Path) -> None:
    target = tmp_path / "long.docx"
    document = Document()
    document.add_paragraph("x" * (MAX_DOCUMENT_TEXT_CHARS + 100))
    document.save(target)

    result = _dispatch(
        Workspace(tmp_path),
        ToolCall(id="call_long", name="read_document", arguments={"path": "long.docx"}),
    )
    payload = json.loads(result.content)
    assert payload["status"] == "completed"
    assert payload["truncated"] is True
    assert payload["truncation_reasons"] == ["text_limit"]
    assert payload["text"].endswith("...[document text truncated]")
    assert len(payload["text"]) == MAX_DOCUMENT_TEXT_CHARS


def test_read_document_rejects_page_options_for_docx_and_bad_ranges(tmp_path: Path) -> None:
    document = Document()
    document.add_paragraph("text")
    document.save(tmp_path / "notes.docx")
    result = _dispatch(
        Workspace(tmp_path),
        ToolCall(
            id="call_docx_page",
            name="read_document",
            arguments={"path": "notes.docx", "start_page": 1},
        ),
    )
    assert json.loads(result.content)["status"] == "invalid_arguments"

    with pytest.raises(ValueError, match="before"):
        ReadDocumentArguments(path="notes.pdf", start_page=2, end_page=1)
    with pytest.raises(ValueError, match="maximum"):
        ReadDocumentArguments(
            path="notes.pdf",
            start_page=1,
            end_page=MAX_DOCUMENT_PAGES + 1,
        )
    with pytest.raises(ValueError, match="maximum"):
        ReadDocumentArguments(path="notes.pdf", end_page=MAX_DOCUMENT_PAGES + 1)


def test_read_document_rejects_path_escape_and_protected_directory(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, protected_paths=[tmp_path / "sessions"])
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "archive.pdf").write_bytes(_pdf_bytes("secret"))

    for path in ("../outside.pdf", "sessions/archive.pdf"):
        result = _dispatch(
            workspace,
            ToolCall(id="call_path", name="read_document", arguments={"path": path}),
        )
        assert result.is_error is True
        assert json.loads(result.content)["status"] == "invalid_path"


def test_read_document_rejects_source_larger_than_budget(tmp_path: Path) -> None:
    (tmp_path / "large.pdf").write_bytes(b"0" * (MAX_DOCUMENT_SOURCE_BYTES + 1))
    result = _dispatch(
        Workspace(tmp_path),
        ToolCall(id="call_large", name="read_document", arguments={"path": "large.pdf"}),
    )
    payload = json.loads(result.content)
    assert payload["status"] == "too_large"
