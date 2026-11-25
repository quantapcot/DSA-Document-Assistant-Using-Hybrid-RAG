"""Verify JSONL snapshots are wired to the actual upload, deletion and TTL paths."""
from io import BytesIO
import json

from docx import Document
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from qdrant_client import QdrantClient

from app.api.routes import documents, sessions, upload
from app.core.config import settings
from app.orchestration import ingestion_flow
from app.orchestration.study_sessions import SessionRegistry
from app.retrieval import indexing
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.vector_store import scope_filter

FILENAME = "Ôn tập đồ thị.docx"


def docx(marker):
    document = Document()
    document.add_paragraph(
        f"{marker} Đồ thị gồm tập các đỉnh và tập các cạnh. Duyệt theo chiều rộng dùng hàng đợi "
        "để lưu các đỉnh sẽ thăm và đánh dấu trước khi đưa vào hàng đợi."
    )
    with BytesIO() as buffer:
        document.save(buffer)
        return buffer.getvalue()


@pytest.fixture
def setup_app(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EXPORT_DEBUG_CHUNKS", True)
    monkeypatch.setattr(indexing, "embedding_dimension", lambda *args, **kwargs: 2)
    monkeypatch.setattr(indexing, "embed_texts", lambda texts, *args, **kwargs: [[1., 0.]] * len(texts))
    clock = [0]
    registry = SessionRegistry(tmp_path, 60, clock=lambda: clock[0])
    client = QdrantClient(":memory:")
    engine = RetrievalEngine(client=client, config=RetrievalConfig(qdrant_mode="memory"))
    app = FastAPI()
    app.state.study_sessions = registry
    app.state.retrieval_engine = engine
    for router in (upload.router, documents.router, sessions.router):
        app.include_router(router)
    try:
        with TestClient(app) as api:
            yield api, registry, engine, clock
    finally:
        client.close()


def session(api):
    response = api.post("/api/sessions")
    response.raise_for_status()
    data = response.json()
    return data["workspace_id"], {"Authorization": f"Bearer {data['token']}"}


def post(api, headers, marker):
    return api.post("/api/upload", headers=headers, files={"file": (FILENAME, docx(marker))})


def assert_snapshot(response, registry, engine, workspace):
    response.raise_for_status()
    document_id = response.json()["document_id"]
    path = registry.root / "debug_chunks" / workspace / f"{document_id}_chunks.jsonl"
    points, _ = engine.client.scroll(engine.config.collection,
                                     scroll_filter=scope_filter("study-sessions", workspace), limit=100,
                                     with_payload=True, with_vectors=False)
    expected = [{key: point.payload[key] for key in ("chunk_id", "document_id", "filename", "page", "section", "text")}
                | {"chunk_index": point.payload["metadata"]["chunk_index"]} for point in points]
    decoded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert decoded == sorted(expected, key=lambda entry: entry["chunk_index"])
    assert len(decoded) == response.json()["indexed_chunks"]
    return path


def test_upload_replacement_failed_index_and_document_delete(setup_app, monkeypatch):
    api, registry, engine, _ = setup_app
    first_workspace, first_headers = session(api)
    second_workspace, second_headers = session(api)
    first = post(api, first_headers, "Bản đầu.")
    first_path = assert_snapshot(first, registry, engine, first_workspace)
    second_path = assert_snapshot(post(api, second_headers, "Phiên khác."), registry, engine, second_workspace)
    previous = first_path.read_bytes()

    replaced = post(api, first_headers, "Bản thay thế.")
    assert replaced.json()["replaced"]
    assert assert_snapshot(replaced, registry, engine, first_workspace) == first_path
    assert first_path.read_bytes() != previous and "Bản thay thế." in first_path.read_text(encoding="utf-8")
    snapshot = first_path.read_bytes()
    with monkeypatch.context() as patch:
        def failed_index(*args, **kwargs):
            raise RuntimeError("fixture index failure")
        patch.setattr(indexing, "embed_texts", failed_index)
        assert post(api, first_headers, "Bản lỗi.").status_code == 500
    assert first_path.read_bytes() == snapshot

    monkeypatch.setattr(settings, "EXPORT_DEBUG_CHUNKS", False)
    deleted = api.delete(f"/api/documents/{first.json()['document_id']}", headers=first_headers)
    assert deleted.status_code == 200 and not first_path.exists() and second_path.is_file()


def test_session_delete_and_expired_session_cleanup(setup_app, monkeypatch):
    api, registry, engine, clock = setup_app
    expired_workspace, expired_headers = session(api)
    expired_path = assert_snapshot(post(api, expired_headers, "Hết hạn."), registry, engine, expired_workspace)
    clock[0] = 30
    current_workspace, current_headers = session(api)
    current_path = assert_snapshot(post(api, current_headers, "Đang chạy."), registry, engine, current_workspace)

    monkeypatch.setattr(settings, "EXPORT_DEBUG_CHUNKS", False)
    clock[0] = 61
    registry.cleanup(engine)
    assert not expired_path.parent.exists() and current_path.is_file()
    assert api.delete("/api/sessions/current", headers=current_headers).status_code == 200
    assert not current_path.parent.exists()


def test_disabling_export_and_export_error_dont_block_upload(setup_app, monkeypatch, caplog):
    api, registry, _, _ = setup_app
    _, headers = session(api)
    monkeypatch.setattr(settings, "EXPORT_DEBUG_CHUNKS", False)
    assert post(api, headers, "Tắt debug.").status_code == 200
    assert not (registry.root / "debug_chunks").exists()

    def denied(*args, **kwargs):
        raise PermissionError("fixture export failure")
    monkeypatch.setattr(settings, "EXPORT_DEBUG_CHUNKS", True)
    monkeypatch.setattr(ingestion_flow, "export_debug_chunks", denied)
    response = post(api, headers, "Bản được lập chỉ mục.")
    assert response.status_code == 200 and response.json()["indexed_chunks"] == 1
    assert api.get("/api/documents", headers=headers).json()["total"] == 1
    assert "Debug chunks export failed: PermissionError" in caplog.text
