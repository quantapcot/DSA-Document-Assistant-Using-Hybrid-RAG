"""Phase 1 regression tests: runtime/upload paths, 15 MB upload limit, and Docker/Qdrant config."""
from __future__ import annotations

from pathlib import Path
import re
import tomllib

import fitz
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.api.routes import documents, upload
from app.core.config import settings
from app.retrieval import indexing
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_valid_pdf_with_min_size(min_bytes: int, keyword: str = "AVL") -> bytes:
    """Create a valid parseable PDF whose byte length is at least min_bytes."""
    with fitz.open() as doc:
        doc.new_page().insert_textbox(
            fitz.Rect(40, 40, 550, 150),
            f"{keyword} tree keeps height balanced using left and right rotations. "
            "Every node tracks a balance factor so lookup, insertion, and deletion "
            "all run in logarithmic time for data structures and algorithms.",
        )
        raw = doc.tobytes()
    if len(raw) >= min_bytes:
        return raw
    # Append PDF comment padding before EOF is avoided; trailing comment after %%EOF is valid in PyMuPDF.
    padding_needed = min_bytes - len(raw)
    return raw + b"\n%" + (b"0" * max(0, padding_needed - 2))


def test_upload_dir_follows_session_runtime_dir_for_local_and_docker(monkeypatch, tmp_path):
    assert settings.upload_dir == Path(settings.SESSION_RUNTIME_DIR) / "uploads"
    assert upload.resolve_upload_dir() == settings.upload_dir

    # Local Path override
    monkeypatch.setattr(settings, "SESSION_RUNTIME_DIR", tmp_path / "local_runtime")
    assert settings.upload_dir == tmp_path / "local_runtime" / "uploads"
    assert upload.resolve_upload_dir() == tmp_path / "local_runtime" / "uploads"

    # Docker string path override (verifies Path conversion type safety)
    monkeypatch.setattr(settings, "SESSION_RUNTIME_DIR", "/app/data/runtime")
    assert settings.upload_dir == Path("/app/data/runtime/uploads")
    assert upload.resolve_upload_dir() == Path("/app/data/runtime/uploads")


def test_upload_and_documents_share_single_runtime_root(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SESSION_RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(indexing, "embedding_dimension", lambda *args, **kwargs: 2)
    monkeypatch.setattr(indexing, "embed_texts", lambda texts, *args, **kwargs: [[1.0, 0.0] for _ in texts])

    client = QdrantClient(":memory:")
    config = RetrievalConfig(qdrant_mode="memory", batch_size=2)
    engine = RetrievalEngine(client=client, config=config, query_embedder=lambda _: [1.0, 0.0])
    app = FastAPI()
    app.state.retrieval_engine = engine
    app.include_router(upload.router)
    app.include_router(documents.router)

    try:
        with TestClient(app) as api:
            pdf_bytes = _make_valid_pdf_with_min_size(1024, keyword="Dijkstra")
            resp = api.post(
                "/api/upload",
                files={"file": ("graph.pdf", pdf_bytes)},
                data={"workspace_id": "phase1-ws", "corpus_id": "user-uploads"},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            expected_uploads_dir = tmp_path / "uploads"
            stored_files = list(expected_uploads_dir.rglob(body["stored_as"]))
            assert len(stored_files) == 1

            del_resp = api.delete(
                f"/api/documents/{body['document_id']}",
                params={"workspace_id": "phase1-ws", "corpus_id": "user-uploads"},
            )
            assert del_resp.status_code == 200, del_resp.text
            assert del_resp.json()["deleted_physical_files"] == [body["stored_as"]]
            assert not stored_files[0].exists()
    finally:
        client.close()


def test_upload_limit_is_synchronized_to_15_mb(monkeypatch):
    config_toml = REPO_ROOT / "frontend" / ".streamlit" / "config.toml"
    parsed = tomllib.loads(config_toml.read_text(encoding="utf-8"))
    # Server transport ceiling includes 1 MiB multipart overhead headroom above exact 15 MiB limit.
    assert parsed["server"]["maxUploadSize"] == 16

    assert upload.MAX_FILE_SIZE == 15 * 1024 * 1024

    upload_ui_src = (REPO_ROOT / "frontend" / "screens" / "upload_ui.py").read_text(encoding="utf-8")
    assert "max_upload_size=15" in upload_ui_src

    monkeypatch.syspath_prepend(str(REPO_ROOT / "frontend"))
    from screens import upload_ui

    assert upload_ui.MAX_UPLOAD_BYTES == 15 * 1024 * 1024


def test_upload_size_boundaries_accept_11_mib_and_reject_over_15_mib(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SESSION_RUNTIME_DIR", tmp_path)
    monkeypatch.setattr(indexing, "embedding_dimension", lambda *args, **kwargs: 2)
    monkeypatch.setattr(indexing, "embed_texts", lambda texts, *args, **kwargs: [[1.0, 0.0] for _ in texts])

    client = QdrantClient(":memory:")
    config = RetrievalConfig(qdrant_mode="memory", batch_size=2)
    engine = RetrievalEngine(client=client, config=config, query_embedder=lambda _: [1.0, 0.0])
    app = FastAPI()
    app.state.retrieval_engine = engine
    app.include_router(upload.router)

    try:
        with TestClient(app) as api:
            # Valid PDF < 10 MB
            small_pdf = _make_valid_pdf_with_min_size(100 * 1024, keyword="Heap")
            r_small = api.post(
                "/api/upload",
                files={"file": ("small.pdf", small_pdf)},
                data={"workspace_id": "limit-ws", "corpus_id": "user-uploads"},
            )
            assert r_small.status_code == 200, r_small.text

            # Valid PDF > 10 MB and <= 15 MiB (11 MiB)
            mid_pdf = _make_valid_pdf_with_min_size(11 * 1024 * 1024, keyword="AVL")
            assert 10 * 1024 * 1024 < len(mid_pdf) <= 15 * 1024 * 1024
            r_mid = api.post(
                "/api/upload",
                files={"file": ("medium_11mib.pdf", mid_pdf)},
                data={"workspace_id": "limit-ws", "corpus_id": "user-uploads"},
            )
            assert r_mid.status_code == 200, r_mid.text

            # Over 15 MiB -> HTTP 413
            oversize = b"%PDF-1.4\n" + (b"x" * (15 * 1024 * 1024))
            r_over = api.post(
                "/api/upload",
                files={"file": ("over_15mib.pdf", oversize)},
                data={"workspace_id": "limit-ws", "corpus_id": "user-uploads"},
            )
            assert r_over.status_code == 413

            # Unsupported format -> HTTP 400
            r_bad_ext = api.post(
                "/api/upload",
                files={"file": ("notes.txt", b"plain text")},
                data={"workspace_id": "limit-ws", "corpus_id": "user-uploads"},
            )
            assert r_bad_ext.status_code == 400
    finally:
        client.close()


def test_docker_compose_qdrant_tag_and_runtime_dir():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    image = compose["services"]["qdrant"]["image"]
    assert image.startswith("qdrant/qdrant:v")
    assert not image.endswith(":latest")
    match = re.search(r":v(\d+)\.(\d+)\.(\d+)$", image)
    assert match is not None, f"Expected pinned semver tag, got {image}"
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (1, 10), f"Qdrant server must support query_points (>=v1.10), got {image}"

    backend_env = compose["services"]["backend"]["environment"]
    assert backend_env["SESSION_RUNTIME_DIR"] == "/app/data/runtime"
    assert (REPO_ROOT / "backend" / ".dockerignore").is_file()
    assert (REPO_ROOT / "frontend" / ".dockerignore").is_file()
