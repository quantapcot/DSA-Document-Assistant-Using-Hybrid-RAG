"""Debug snapshots are UTF-8, atomic and contained in the selected workspace."""
import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.orchestration.debug_chunks import export_debug_chunks, remove_debug_document, remove_debug_workspace
from app.schemas.contracts import DocumentChunk


def chunk(text, index=0):
    return DocumentChunk(
        chunk_id=f"chunk-{index}", document_id="doc", filename="Đồ thị.docx", page=index + 1,
        section="Cấu trúc đồ thị" if index == 0 else None, text=text,
        metadata={"chunk_index": index, "embedding": [1.0], "session_token": "fixture-secret"},
    )


def test_export_utf8_one_json_object_per_chunk(tmp_path):
    chunks = [chunk("Đồ thị có đỉnh và cạnh.\nVí dụ có hướng."), chunk("Duyệt theo chiều rộng.", 1)]
    path = export_debug_chunks(tmp_path, "workspace", "doc", chunks)
    assert path == tmp_path / "debug_chunks" / "workspace" / "doc_chunks.jsonl"
    raw = path.read_bytes()
    assert "Đồ thị".encode() in raw and b"fixture-secret" not in raw
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert [json.loads(line) for line in raw.decode("utf-8").splitlines()] == [
        {"chunk_id": item.chunk_id, "document_id": item.document_id, "filename": item.filename,
         "page": item.page, "section": item.section, "text": item.text, "chunk_index": index}
        for index, item in enumerate(chunks)
    ]


def test_export_replaces_whole_snapshot_and_preserves_it_on_failed_publish(tmp_path, monkeypatch):
    path = export_debug_chunks(tmp_path, "workspace", "doc", [chunk("first"), chunk("second", 1)])
    assert export_debug_chunks(tmp_path, "workspace", "doc", [chunk("new")]) == path
    previous = path.read_bytes()
    assert [json.loads(line)["text"] for line in previous.splitlines()] == ["new"]

    def denied(self, target):
        raise PermissionError("fixture export failure")

    monkeypatch.setattr(Path, "replace", denied)
    with pytest.raises(PermissionError):
        export_debug_chunks(tmp_path, "workspace", "doc", [chunk("not published")])
    assert path.read_bytes() == previous
    assert list(path.parent.iterdir()) == [path]


def test_cleanup_is_idempotent_and_keeps_other_workspaces(tmp_path):
    a = export_debug_chunks(tmp_path, "workspace-a", "doc", [chunk("a")])
    b = export_debug_chunks(tmp_path, "workspace-b", "doc", [chunk("b")])
    remove_debug_document(tmp_path, "workspace-a", "doc")
    remove_debug_document(tmp_path, "workspace-a", "doc")
    assert not a.exists() and b.is_file()
    remove_debug_workspace(tmp_path, "workspace-a")
    remove_debug_workspace(tmp_path, "workspace-a")
    assert not a.parent.exists() and b.is_file()


@pytest.mark.parametrize(("workspace", "document"), [
    ("..", "doc"), ("../outside", "doc"), ("safe", r"..\outside"), ("safe", "NUL"),
])
def test_debug_paths_cannot_escape_runtime(tmp_path, workspace, document):
    with pytest.raises(ValueError):
        export_debug_chunks(tmp_path, workspace, document, [chunk("text")])
    assert not (tmp_path / "debug_chunks").exists()


def test_debug_env_flag_is_opt_in(monkeypatch):
    monkeypatch.delenv("EXPORT_DEBUG_CHUNKS", raising=False)
    assert Settings(_env_file=None).EXPORT_DEBUG_CHUNKS is False
    monkeypatch.setenv("EXPORT_DEBUG_CHUNKS", "true")
    assert Settings(_env_file=None).EXPORT_DEBUG_CHUNKS is True
