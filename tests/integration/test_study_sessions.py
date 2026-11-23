"""Exercise anonymous session ownership, persisted TTL, and scoped cleanup."""
from io import BytesIO
from threading import Event

from docx import Document
from fastapi.testclient import TestClient
import fitz
import pytest
from qdrant_client import QdrantClient

from app.generation import rag_engine
from app import main as backend
from app.main import app
from app.orchestration.study_sessions import SESSION_CORPUS, SessionRegistry
from app.retrieval import embeddings as embedding_models, indexing, reranker
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.vector_store import scope_filter

GRAPH = "Breadth-first search uses a FIFO queue to traverse a graph by increasing distance from the starting vertex."
SORT = "Merge sort divides the input array into smaller parts, sorts each part, and combines them into a sorted result."
EVAL = ("dsa-eval-v1", "evaluation")
LEGACY = ("user-uploads", "legacy-workspace")


@pytest.fixture(autouse=True)
def embeddings(monkeypatch):
    monkeypatch.setattr(indexing, "embedding_dimension", lambda *args, **kwargs: 2)
    monkeypatch.setattr(indexing, "embed_texts", lambda texts, *args, **kwargs: [[1., 0.]] * len(texts))


def bind_backend(monkeypatch, client, registry):
    cfg = RetrievalConfig(qdrant_mode="memory")
    engine = RetrievalEngine(config=cfg, client=client, query_embedder=lambda _: [1., 0.],
                             rerank_scorer=lambda pairs: [.95] * len(pairs))
    monkeypatch.setitem(app.state._state, "retrieval_engine", engine)
    monkeypatch.setitem(app.state._state, "study_sessions", registry)
    return engine, TestClient(app)


def session(api):
    response = api.post("/api/sessions")
    assert response.status_code == 201 and response.headers["Cache-Control"] == "no-store"
    data = response.json()
    return data, {"Authorization": f"Bearer {data['token']}"}


def upload(api, headers, filename, body, data=None):
    return api.post("/api/upload", headers=headers, files={"file": (filename, body)}, data=data)


def pdf(text):
    with fitz.open() as doc:
        doc.new_page().insert_textbox(fitz.Rect(50, 50, 540, 140), text)
        return doc.tobytes()


def docx(text):
    with BytesIO() as buffer:
        doc = Document()
        doc.add_paragraph(text)
        doc.save(buffer)
        return buffer.getvalue()


def names(api, headers):
    response = api.get("/api/documents", headers=headers)
    assert response.status_code == 200, response.text
    return {document["filename"] for document in response.json()["documents"]}


def seed(engine, scope):
    corpus, workspace = scope
    indexing.build_index([{
        "chunk_id": "seed", "document_id": "seed", "filename": "seed.pdf", "page": 1, "text": GRAPH,
    }], client=engine.client, config=engine.config, corpus_id=corpus, workspace_id=workspace)


def count(engine, scope):
    return engine.client.count(engine.config.collection, count_filter=scope_filter(*scope), exact=True).count


def test_token_isolates_pdf_docx_and_chat_and_limits_deletion(tmp_path, monkeypatch, caplog):
    registry = SessionRegistry(tmp_path / "runtime", 300)
    client = QdrantClient(":memory:")
    engine, api = bind_backend(monkeypatch, client, registry)
    try:
        seed(engine, EVAL)
        seed(engine, LEGACY)
        a, auth_a = session(api)
        b, auth_b = session(api)
        first_id, second_id = a["workspace_id"], b["workspace_id"]
        assert first_id != second_id and a["token"] != b["token"]
        assert names(api, auth_a) == names(api, auth_b) == set()
        with registry.database() as db:
            assert all(value not in (a["token"], b["token"])
                       for value, in db.execute("SELECT token_hash FROM study_sessions"))

        first = upload(api, auth_a, "Graphs.pdf", pdf(GRAPH))
        second = upload(api, auth_b, "Sorting.docx", docx(SORT))
        assert first.status_code == second.status_code == 200
        assert names(api, auth_a) == {"Graphs.pdf"} and names(api, auth_b) == {"Sorting.docx"}
        registry_dir = registry.directory(first_id)
        assert len(list(registry_dir.rglob("*.pdf"))) == 1

        def chat(system, prompt, **kwargs):
            assert "Graphs.pdf" in prompt and "Sorting.docx" not in prompt and "seed.pdf" not in prompt
            return "BFS sử dụng Queue [1]."

        monkeypatch.setattr(rag_engine, "generate_answer", chat)
        answer = api.post("/api/chat", headers=auth_a, json={"query": "BFS uses a queue?"}).json()
        assert answer["generation_mode"] == "llm" and answer["citations"][0]["filename"] == "Graphs.pdf"
        assert api.post("/api/quiz", headers=auth_b, json={"topic": "BFS", "num_questions": 1}).json()["questions"] == []
        stolen = {"workspace_id": first_id, "corpus_id": SESSION_CORPUS}
        assert api.get("/api/documents", params=stolen).status_code == 401
        assert api.get("/api/documents", params=stolen, headers=auth_b).status_code == 403
        for path, body in (("chat", {"query": "BFS"}), ("quiz", {"topic": "BFS"}), ("roadmap", {})):
            assert api.post(f"/api/{path}", headers=auth_b, json={**stolen, **body}).status_code == 403
        assert upload(api, auth_b, "bad.pdf", pdf(GRAPH), stolen).status_code == 403
        assert api.delete(f"/api/documents/{first.json()['document_id']}", headers=auth_b).status_code == 404
        assert api.delete("/api/sessions/current", headers=auth_b, params=stolen).status_code == 403
        assert api.delete("/api/sessions/current").status_code == 401
        cleared = api.delete("/api/sessions/current", headers=auth_a)
        assert cleared.status_code == 200 and not registry_dir.exists()
        assert api.get("/api/documents", headers=auth_a).status_code == 401
        assert count(engine, (SESSION_CORPUS, first_id)) == 0
        assert names(api, auth_b) == {"Sorting.docx"}
        assert count(engine, EVAL) == count(engine, LEGACY) == 1
        assert a["token"] not in caplog.text and b["token"] not in caplog.text
    finally:
        client.close()


def test_restart_reopens_disk_registry_and_drops_only_expired_scopes(tmp_path, monkeypatch):
    now = [100.0]
    root = tmp_path / "runtime"
    database = str(tmp_path / "qdrant")
    registry = SessionRegistry(root, 10, clock=lambda: now[0])
    client = QdrantClient(path=database)
    engine, api = bind_backend(monkeypatch, client, registry)
    try:
        seed(engine, EVAL)
        expired, old_auth = session(api)
        assert upload(api, old_auth, "A.pdf", pdf(GRAPH)).status_code == 200
        now[0] = 105
        active, new_auth = session(api)
        assert upload(api, new_auth, "B.docx", docx(SORT)).status_code == 200
        old_dir = registry.directory(expired["workspace_id"])
    finally:
        client.close()

    now[0] = 111
    registry = SessionRegistry(root, 10, clock=lambda: now[0])
    client = QdrantClient(path=database)
    engine, api = bind_backend(monkeypatch, client, registry)
    assert old_dir.exists() and api.get("/api/documents", headers=old_auth).status_code == 401
    completed = Event()
    cleanup = registry.cleanup

    def sweep(retrieval):
        cleanup(retrieval)
        completed.set()

    monkeypatch.setattr(registry, "cleanup", sweep)
    monkeypatch.setattr(backend, "RetrievalEngine", lambda **kwargs: engine)
    monkeypatch.setattr(embedding_models, "_get_model", lambda *args: None)
    monkeypatch.setattr(reranker, "_get_model", lambda *args: None)
    with TestClient(app) as restarted:
        assert completed.wait(5)
        assert not old_dir.exists()
        assert count(engine, (SESSION_CORPUS, expired["workspace_id"])) == 0
        assert names(restarted, new_auth) == {"B.docx"} and count(engine, EVAL) == 1
        assert registry.authenticate(active["token"]) == active["workspace_id"]


def test_failed_session_deletion_revokes_token_and_retries_from_tombstone(tmp_path, monkeypatch):
    registry = SessionRegistry(tmp_path / "runtime", 300)
    client = QdrantClient(":memory:")
    engine, api = bind_backend(monkeypatch, client, registry)
    try:
        data, headers = session(api)
        assert upload(api, headers, "A.pdf", pdf(GRAPH)).status_code == 200
        directory = registry.directory(data["workspace_id"])
        with monkeypatch.context() as ctx:
            def bad_delete(**kwargs):
                raise OSError("temporary outage")
            ctx.setattr(client, "delete", bad_delete)
            assert api.delete("/api/sessions/current", headers=headers).status_code == 202
        assert directory.exists() and api.get("/api/documents", headers=headers).status_code == 401
        registry.cleanup(engine)
        assert not directory.exists() and not registry.pending_cleanup()
    finally:
        client.close()
