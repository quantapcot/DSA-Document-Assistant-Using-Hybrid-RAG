"""Production authentication and explicit Docker config fail-safes."""
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.core.config import settings
from app.main import app, load_retrieval_config
from app.orchestration.study_sessions import SessionRegistry


@pytest.fixture
def production_api(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ENV", "production")
    registry = SessionRegistry(tmp_path, 3600)
    mock_client = SimpleNamespace(collection_exists=lambda _: False,
                                  get_collections=lambda: [])
    engine = SimpleNamespace(client=mock_client,
                             config=SimpleNamespace(collection="dsa_chunks_v1",
                                                    reranker_model="BAAI/bge-reranker-v2-m3"))
    monkeypatch.setitem(app.state._state, "study_sessions", registry)
    monkeypatch.setitem(app.state._state, "retrieval_engine", engine)
    return TestClient(app), registry, mock_client


def test_production_rejects_all_unauthenticated_legacy_study_paths(production_api):
    api, _, _ = production_api
    assert api.get("/api/documents").status_code == 401
    assert api.get("/api/documents", params={"workspace_id": "public-workspace"}).status_code == 401
    assert api.post("/api/chat", json={"query": "BFS"}).status_code == 401
    assert api.post("/api/quiz", json={"topic": "BFS"}).status_code == 401
    assert api.post("/api/roadmap", json={"topic": "Graph"}).status_code == 401
    assert api.delete("/api/documents/dummy").status_code == 401
    assert api.post("/api/upload", files={"file": ("fake.pdf", b"%PDF-1.4")}).status_code == 401


def test_production_session_token_remains_usable(production_api):
    api, registry, _ = production_api
    session = registry.create()
    headers = {"Authorization": f"Bearer {session['token']}"}
    response = api.get("/api/documents", headers=headers)
    assert response.status_code == 200 and response.json()["total"] == 0
    assert api.get("/api/documents", headers={"Authorization": "Bearer invalid"}).status_code == 401


def test_readiness_checks_actual_qdrant_connectivity(production_api):
    api, _, mock_client = production_api
    response = api.get("/ready")
    assert response.status_code == 200 and response.json()["reranker_configured"] is True

    def broken():
        raise ConnectionError("test connection failure")

    mock_client.get_collections = broken
    assert api.get("/ready").status_code == 503


def test_docker_yaml_config_honors_qdrant_environment(tmp_path, monkeypatch):
    yaml_path = tmp_path / "retrieval.yaml"
    yaml_path.write_text("qdrant_mode: local\nreranker_model: BAAI/bge-reranker-v2-m3\n", encoding="utf-8")
    monkeypatch.setenv("RETRIEVAL_CONFIG_PATH", str(yaml_path))
    monkeypatch.setenv("QDRANT_MODE", "server")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    cfg = load_retrieval_config()
    assert cfg.qdrant_mode == "server"
    assert cfg.qdrant_url == "http://qdrant:6333"
    assert cfg.reranker_model == "BAAI/bge-reranker-v2-m3"
    monkeypatch.setenv("RETRIEVAL_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        load_retrieval_config()
