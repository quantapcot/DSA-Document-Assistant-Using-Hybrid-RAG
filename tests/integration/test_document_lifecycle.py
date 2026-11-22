from io import BytesIO

import fitz
import pytest
from docx import Document
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.api.routes import documents, upload
from app.retrieval import indexing
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.vector_store import scope_filter


def pdf(pages, keyword):
    with fitz.open() as document:
        for _ in range(pages):
            document.new_page().insert_textbox(
                fitz.Rect(40, 40, 550, 150),
                f"{keyword} traversal visits each vertex in a graph. "
                "A queue keeps track of pending nodes and prevents repeated visits. "
                "When a node is processed, the search inspects each adjacent vertex.",
            )
        return document.tobytes()


@pytest.fixture(params=["memory", "local"])
def lifecycle(request, tmp_path, monkeypatch):
    monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(indexing, "embedding_dimension", lambda *args, **kwargs: 2)
    monkeypatch.setattr(indexing, "embed_texts",
                        lambda texts, *args, **kwargs: [[1.0, 0.0] for _ in texts])
    mode = request.param
    client = (QdrantClient(":memory:") if mode == "memory"
              else QdrantClient(path=str(tmp_path / "qdrant")))
    config = RetrievalConfig(qdrant_mode=mode, batch_size=2)
    engine = RetrievalEngine(client=client, config=config, query_embedder=lambda _: [1.0, 0.0])
    app = FastAPI()
    app.state.retrieval_engine = engine
    app.include_router(upload.router)
    app.include_router(documents.router)
    with TestClient(app) as api:
        yield api, engine, tmp_path
    client.close()


def post(api, workspace, filename, content, corpus="user-uploads"):
    return api.post("/api/upload", files={"file": (filename, content)},
                    data={"workspace_id": workspace, "corpus_id": corpus})


def query(engine, word, workspace, method="bm25"):
    return engine.retrieve(word, method, top_k=20, workspace_id=workspace,
                           corpus_id="user-uploads")


def test_pdf_replace_abort_and_delete_across_workspaces(lifecycle, monkeypatch):
    api, engine, root = lifecycle
    first = post(api, "a", "test.pdf", pdf(10, "alphagraph"))
    assert first.status_code == 200, first.text
    original = first.json()
    assert original["indexed_chunks"] == 10
    assert len(query(engine, "alphagraph", "a")) == 10
    assert post(api, "b", "test.pdf", pdf(1, "betagraph")).status_code == 200
    assert post(api, "a", "test.pdf", pdf(1, "gammagraph"), corpus="other").status_code == 200
    replacement = post(api, "a", "test.pdf", pdf(6, "replacementgraph"))
    assert replacement.status_code == 200, replacement.text
    replaced = replacement.json()
    assert replaced["document_id"] == original["document_id"]
    assert replaced["replaced"] and replaced["scope_count"] == 6
    assert not list(root.rglob(original["stored_as"]))
    assert query(engine, "alphagraph", "a") == []
    evidence = query(engine, "replacementgraph", "a")
    assert len(evidence) == 6 and all(c["filename"] == "test.pdf" for c in evidence)

    def fail_second_batch(texts, *args, **kwargs):
        if getattr(fail_second_batch, "called", False):
            raise RuntimeError("embedding failure")
        fail_second_batch.called = True
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(indexing, "embed_texts", fail_second_batch)
    assert post(api, "a", "test.pdf", pdf(4, "failedgraph")).status_code == 500
    assert query(engine, "replacementgraph", "a") == evidence
    assert len(list(root.rglob("*.pdf"))) == 3

    listing = api.get("/api/documents", params={"workspace_id": "a"}).json()["documents"]
    assert listing[0]["chunk_count"] == 6
    assert listing[0]["uploaded_at"] and listing[0]["size_kb"] > 0
    doc_id = replaced["document_id"]
    deleted = api.delete(f"/api/documents/{doc_id}", params={"workspace_id": "a"})
    assert deleted.json()["deleted_chunks"] == 6
    assert query(engine, "replacementgraph", "a") == []
    assert len(query(engine, "betagraph", "b")) == 1
    assert len(list(root.rglob("*.pdf"))) == 2
    assert api.get("/api/documents", params={"workspace_id": "a", "corpus_id": "other"}).json()["total"] == 1


def test_real_docx_and_empty_workspace_with_benchmark_present(lifecycle, monkeypatch):
    api, engine, _ = lifecycle
    document = Document()
    document.add_paragraph(
        "Binary search finds a value in a sorted array by repeatedly narrowing the search range. "
        "The algorithm compares the target with the middle element and continues on one side."
    )
    data = BytesIO()
    document.save(data)
    response = post(api, "a", "notes.docx", data.getvalue())
    assert response.status_code == 200, response.text
    assert query(engine, "binary search", "a", "dense")[0]["filename"] == "notes.docx"
    points, _ = engine.client.scroll(engine.config.collection,
                                     scroll_filter=scope_filter("user-uploads", "a"))
    indexing.build_index([points[0].payload], client=engine.client, config=engine.config,
                         workspace_id="research", corpus_id="benchmark-v1")
    monkeypatch.setattr(engine, "query_embedder", lambda _: pytest.fail("empty scope embedded a query"))
    for method in ("bm25", "dense", "hybrid_rrf", "hybrid_rerank"):
        assert query(engine, "binary search", "empty", method) == []
    assert post(api, "research", "test.pdf", pdf(1, "test"), "benchmark-v1").status_code == 403
    assert api.delete("/api/documents/anything", params={
        "workspace_id": "research", "corpus_id": "benchmark-v1",
    }).status_code == 403


def test_bad_upload_and_unavailable_engine(lifecycle):
    api, engine, root = lifecycle
    assert post(api, "a", "empty.pdf", b"").status_code == 400
    assert post(api, "a", "test.txt", b"hello").status_code == 400
    assert post(api, "a", "bad.docx", b"not a zip").status_code == 400
    api.app.state.retrieval_engine = None
    assert post(api, "a", "test.pdf", pdf(1, "test")).status_code == 503
    assert not engine.client.collection_exists(engine.config.collection)
    assert not list(root.rglob("*.pdf"))


def test_qdrant_write_failure_restores_previous_revision(lifecycle, monkeypatch):
    api, engine, root = lifecycle
    initial = post(api, "a", "test.pdf", pdf(3, "oldgraph"))
    assert initial.status_code == 200, initial.text
    expected = query(engine, "oldgraph", "a")

    real_upsert = engine.client.upsert
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("write failure after first batch")
        return real_upsert(*args, **kwargs)

    monkeypatch.setattr(engine.client, "upsert", fail_once)
    assert post(api, "a", "test.pdf", pdf(4, "newgraph")).status_code == 500
    assert query(engine, "oldgraph", "a") == expected
    assert query(engine, "newgraph", "a") == []
    assert [p.name for p in root.rglob("*.pdf")] == [initial.json()["stored_as"]]


def test_reupload_replaces_legacy_uuid_prefixed_document(lifecycle):
    api, engine, root = lifecycle
    prefix = "f" * 32
    old_filename = f"{prefix}_notes.pdf"
    old_file = root / old_filename
    old_file.write_bytes(pdf(1, "legacygraph"))
    indexing.build_index([{
        "chunk_id": "legacy-chunk", "document_id": f"{prefix}_notes",
        "filename": old_filename, "text": "legacygraph traversal",
        "page": 1, "metadata": {},
    }], client=engine.client, config=engine.config,
        corpus_id="user-uploads", workspace_id="a")
    result = post(api, "a", "notes.pdf", pdf(1, "currentgraph"))
    assert result.status_code == 200, result.text
    assert result.json()["replaced"] and result.json()["scope_count"] == 1
    assert not old_file.exists()
    assert query(engine, "legacygraph", "a") == []
