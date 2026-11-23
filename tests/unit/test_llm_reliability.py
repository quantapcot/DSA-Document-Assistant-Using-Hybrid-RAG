"""Mock provider tests for timing, error classification and safe fallback."""
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings
from app.generation import llm_client, rag_engine
from app.main import app
from app.orchestration import request_flow


class ProviderError(Exception):
    def __init__(self, status, body):
        self.status_code, self.body = status, body
        super().__init__("private key marker")


@pytest.fixture
def configure(monkeypatch):
    settings = Settings(_env_file=None, LLM_PROVIDER="openai", OPENAI_API_KEY=SecretStr("test-key"),
                        LLM_BASE_URL="https://fit.invalid/v1", LLM_MODEL="mock-qwen-27b",
                        LLM_CHAT_TIMEOUT=2, LLM_QUIZ_GENERATION_TIMEOUT=3,
                        LLM_QUIZ_VERIFICATION_TIMEOUT=4, LLM_REVISION_TIMEOUT=5)
    monkeypatch.setattr(llm_client, "settings", settings)
    monkeypatch.setattr(rag_engine, "settings", settings)
    return settings


@pytest.mark.parametrize(("task", "timeout"), [
    ("chat", 2), ("quiz_generation", 3), ("quiz_verification", 4), ("revision", 5),
])
def test_timeout_per_task_and_remaining_budget(configure, task, timeout, monkeypatch):
    call = MagicMock(return_value="ok")
    monkeypatch.setattr(llm_client, "_call_openai", call)
    assert llm_client.generate_answer("system", "prompt", task=task) == "ok"
    assert call.call_args.kwargs["timeout"] == timeout
    token = llm_client.request_deadline.set(time.monotonic() + 0.8)
    try:
        assert llm_client.generate_answer("system", "prompt", task=task) == "ok"
        assert 0 < call.call_args.kwargs["timeout"] <= 0.8
    finally:
        llm_client.request_deadline.reset(token)


@pytest.mark.parametrize(("status", "body", "expected", "attempts"), [
    (429, {"error": {"code": "rate_limit_exceeded"}}, "rate_limit", 3),
    (429, "quota exceeded for requests per minute", "rate_limit", 3),
    (429, {"error": {"code": "insufficient_quota"}}, "quota_exceeded", 1),
    (429, "daily budget exhausted", "quota_exceeded", 1),
    (401, "invalid API key", "authentication", 1),
    (503, "upstream error", "provider", 1),
    (504, "gateway timeout", "timeout", 3),
])
def test_error_classification_and_backoff(configure, status, body, expected, attempts, monkeypatch, caplog):
    call = MagicMock(side_effect=ProviderError(status, body))
    wait = MagicMock()
    monkeypatch.setattr(llm_client, "_call_openai", call)
    monkeypatch.setattr(llm_client.time, "sleep", wait)
    with pytest.raises(llm_client.LLMError) as failure:
        llm_client.generate_answer("system", "prompt", raise_errors=True)
    assert failure.value.code == expected and call.call_count == attempts
    assert [args.args[0] for args in wait.call_args_list] == ([1, 2] if attempts == 3 else [])
    assert "private key marker" not in caplog.text and "test-key" not in str(failure.value)


def test_expired_deadline_prevents_retry(configure, monkeypatch):
    call = MagicMock(side_effect=ProviderError(429, "rate limit"))
    monkeypatch.setattr(llm_client, "_call_openai", call)
    token = llm_client.request_deadline.set(time.monotonic() + 0.5)
    try:
        with pytest.raises(llm_client.LLMError) as failure:
            llm_client.generate_answer("system", "prompt", raise_errors=True)
        assert failure.value.code == "timeout" and call.call_count == 1
    finally:
        llm_client.request_deadline.reset(token)


def test_provider_truncation_and_json_errors(configure, monkeypatch):
    sdk = MagicMock()
    client = sdk.return_value.__enter__.return_value
    client.chat.completions.create.return_value = SimpleNamespace(choices=[
        SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"slot":')),
    ])
    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=sdk))
    with pytest.raises(llm_client.LLMError) as error:
        rag_engine.generate_study_json("topic", [], "output", task="revision", max_tokens=6000)
    assert error.value.code == "truncated_json"
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 4000
    with pytest.raises(llm_client.LLMError) as error:
        monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: "{")
        rag_engine.generate_study_json("topic", [], "output", task="revision")
    assert error.value.code == "invalid_json"


def test_http_deadline_covers_retrieval_and_matches_health(monkeypatch):
    monkeypatch.setattr(request_flow.settings, "CHAT_REQUEST_TIMEOUT", 0.02)
    def slow_retrieve(**kwargs):
        time.sleep(0.15)
        return []
    monkeypatch.setattr(app.state, "retrieval_engine", SimpleNamespace(retrieve=slow_retrieve), raising=False)
    api = TestClient(app)
    assert api.get("/health").json()["study_request_timeouts"]["chat"] == 0.02
    response = api.post("/api/chat", json={"query": "graph"})
    assert response.status_code == 504 and response.json()["detail"]["code"] == "timeout"


def test_frontend_uses_health_deadline_and_displays_backend_timeout(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "frontend"))
    import api_client

    monkeypatch.setattr(api_client, "_STUDY_TIMEOUTS", api_client._STUDY_TIMEOUTS.copy())
    monkeypatch.setattr(request_flow.settings, "QUIZ_REQUEST_TIMEOUT", 37)
    backend = TestClient(app)
    health = backend.get("/health")
    get = MagicMock(return_value=SimpleNamespace(ok=health.status_code == 200, json=health.json))
    monkeypatch.setattr(api_client.requests, "get", get)
    message = llm_client.ERROR_MESSAGES["timeout"]
    post = MagicMock(return_value=SimpleNamespace(
        ok=False, status_code=504, json=lambda: {"detail": {"code": "timeout", "message": message}},
    ))
    monkeypatch.setattr(api_client.requests, "post", post)

    assert api_client.check_health()[0]
    assert api_client.create_quiz("Graph traversal", workspace_id="student") == (
        False, {"error": message},
    )
    assert post.call_args.kwargs["timeout"] == (10, 47)
    assert post.call_args.kwargs["json"] == {
        "topic": "Graph traversal", "num_questions": 5, "workspace_id": "student",
    }
