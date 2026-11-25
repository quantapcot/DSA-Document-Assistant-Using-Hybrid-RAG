import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from qdrant_client import QdrantClient

from app.core.config import Settings
from app.generation import llm_client, rag_engine
from app.generation.abstention import should_abstain
from app.generation.quiz_generator import generate_quiz
from app.main import app


@pytest.fixture
def evidence():
    return [
        {"filename": "trees.pdf", "page": 7, "text": "Binary search compares the target against a sorted collection.", "score": .025},
        {"filename": "graphs.pdf", "page": 9, "text": "Graph traversal visits adjacent vertices using a queue or stack.", "score": .020},
    ]


@pytest.mark.parametrize(("method", "score", "abstains"), [
    ("hybrid_rrf", .025, False), ("hybrid_rrf", .005, True),
    ("hybrid_rerank", .15, True), ("hybrid_rerank", .25, False),
    ("dense", .40, False), ("bm25", 4.99, True),
])
def test_method_specific_abstention(method, score, abstains):
    assert should_abstain([{"score": score}], method) is abstains
    assert should_abstain([], method)


def test_confidence_depends_on_citation_coverage_and_relevance(evidence):
    calculate = rag_engine.calculate_answer_confidence
    method = "hybrid_rrf"
    weak = calculate("Answer [2]", evidence, method)
    strong = calculate("Answer [1]", evidence, method)
    full = calculate("Answer [1] [2]", evidence, method)
    assert .1 <= weak < strong < full <= .98
    assert calculate("Có thể là [1] [2]", evidence, method) < full
    assert calculate("No citations [99]", evidence, method) == .1


def test_chat_retains_original_citation_labels(monkeypatch, evidence):
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: "Graph [2], [2], [99].")
    result = rag_engine.build_chat_response("graph", evidence, method_used="hybrid_rrf")
    assert result.generation_mode == "llm"
    assert [(c.label, c.filename) for c in result.citations] == [(2, "graphs.pdf")]


def test_chat_preserves_provider_abstention(monkeypatch, evidence):
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: rag_engine.NO_EVIDENCE_MESSAGE)
    result = rag_engine.build_chat_response("unrelated", evidence, method_used="hybrid_rrf")
    assert result.generation_mode == "abstained" and not result.citations


def test_extraction_when_key_missing(monkeypatch, evidence):
    monkeypatch.setattr(llm_client, "settings", Settings(
        _env_file=None, GROQ_API_KEY=None,
    ))
    chat = rag_engine.build_chat_response("search", evidence, method_used="hybrid_rrf")
    assert chat.generation_mode == "extractive" and len(chat.citations) == 2
    quiz = generate_quiz("search", evidence, num_questions=4)
    assert quiz.generation_mode == "unavailable" and not quiz.questions


def test_gateway_loaded_from_dotenv_and_secret_masked(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("GROQ_BASE_URL=https://school.invalid/v1\nGROQ_API_KEY=mock-secret\n")
    for name in ("GROQ_BASE_URL", "GROQ_API_KEY", "GROQ_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(_env_file=env)
    assert isinstance(settings.GROQ_API_KEY, SecretStr)
    assert "mock-secret" not in repr(settings)
    constructor = MagicMock()
    client = constructor.return_value.__enter__.return_value
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer."))]
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))
    monkeypatch.setattr(llm_client, "settings", settings)
    assert llm_client.generate_answer("system", "question") == "Grounded answer."
    assert constructor.call_args.kwargs["base_url"] == "https://school.invalid/v1"


@pytest.mark.parametrize("failure", [TimeoutError("test-key"), type("RateLimit", (Exception,), {"status_code": 429})("test-key")])
def test_llm_retries_without_logging_key(monkeypatch, caplog, failure):
    monkeypatch.setattr(llm_client, "settings", Settings(
        _env_file=None, GROQ_API_KEY="test-key",
    ))
    call = MagicMock(side_effect=failure)
    sleep = MagicMock()
    monkeypatch.setattr(llm_client, "_call_groq", call)
    monkeypatch.setattr(llm_client.time, "sleep", sleep)
    assert llm_client.generate_answer("system", "question") == ""
    assert call.call_count == 3
    assert [args.args[0] for args in sleep.call_args_list] == [1, 2]
    assert "test-key" not in caplog.text


QUIZ = {
    "question": "What is compared?", "options": ["target", "root", "edge", "height"],
    "correct_index": 0, "explanation": "The target is compared.", "sources": [1],
    "topic": "Binary search",
}


def test_quiz_api_uses_mock_llm_and_keeps_scope(monkeypatch, evidence):
    reranked = [{**item, "score": .8} for item in evidence]
    retrieve = MagicMock(side_effect=[RuntimeError("CUDA OOM"), evidence, *([reranked] * 3)])
    monkeypatch.setattr(app.state, "retrieval_engine", SimpleNamespace(retrieve=retrieve), raising=False)
    verified = [{"slot": 1, "correct_index": 0, "unique_answer": True, "on_topic": True,
                 "self_contained": True, "vietnamese": True, "source": 1, "quote": evidence[0]["text"]}]
    from app.generation import quiz_generator
    monkeypatch.setattr(quiz_generator, "MAX_PASSES", 2)
    llm = MagicMock(side_effect=[
        f"```json\n{json.dumps([{**QUIZ, 'slot': 1}])}\n```", json.dumps(verified),
        json.dumps([{"slot": 1, "supported": True, "distinct": True}]), "[]",
    ])
    monkeypatch.setattr(rag_engine, "generate_answer", llm)
    api = TestClient(app)
    quiz = api.post("/api/quiz", json={"topic": "Binary search", "workspace_id": "student"})
    assert quiz.status_code == 200
    assert quiz.json()["questions"][0] == {**QUIZ, "evidence": quiz.json()["citations"]}
    assert quiz.json()["generation_mode"] == "llm"
    assert [call.kwargs["method"] for call in retrieve.call_args_list] == [
        "hybrid_rerank", "hybrid_rrf", "hybrid_rerank",
    ]
    assert all(call.kwargs["workspace_id"] == "student"
               and call.kwargs["corpus_id"] == "user-uploads" for call in retrieve.call_args_list)


def test_empty_workspace_never_queries_benchmark(monkeypatch):
    client = QdrantClient(":memory:")
    retrieve = MagicMock(return_value=[])
    engine = SimpleNamespace(retrieve=retrieve, client=client, config=SimpleNamespace(collection="test"))
    monkeypatch.setattr(app.state, "retrieval_engine", engine, raising=False)
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: pytest.fail("called model"))
    try:
        api = TestClient(app)
        for path in ("chat", "quiz", "roadmap"):
            body = {"topic": "tree"} if path == "roadmap" else {"query": "tree"}
            response = api.post(f"/api/{path}", json=body)
            assert response.status_code == 200
            data = response.json()
            assert "Workspace chưa có tài liệu" in data["answer"]
            assert data["generation_mode"] == "abstained" and not data["citations"]
        assert retrieve.call_count == 1  # The exercise routes short-circuit empty scopes.
        assert all(call.kwargs["workspace_id"] == "default-workspace"
                   and call.kwargs["corpus_id"] == "user-uploads" for call in retrieve.call_args_list)
    finally:
        client.close()


def test_streamlit_chat_quiz_and_revision_with_mock_api(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "frontend"))
    import api_client
    from screens import chat_ui, quiz_ui, revision_ui

    citation = {"label": 2, "filename": "graphs.pdf", "page": 9, "snippet": "Use a queue."}
    data = {"generation_mode": "extractive", "citations": [citation], "confidence_score": .56}
    monkeypatch.setattr(api_client, "check_health", lambda: (True, {}))
    monkeypatch.setattr(api_client, "create_session", lambda: (True, {
        "workspace_id": "w", "corpus_id": "c", "token": "fixture", "expires_at": float("inf"),
    }))
    monkeypatch.setattr(chat_ui, "ask_question", lambda *args, **kwargs: (True, {**data, "answer": "Use a queue [2]."}))
    monkeypatch.setattr(quiz_ui, "create_quiz", lambda *args, **kwargs: (True, {**data, "questions": [QUIZ]}))
    item = {
        "id": "T1", "title": "Graphs", "objective": "Hiểu BFS", "key_knowledge": ["Queue"],
        "prerequisites": [], "sources": [2], "support_quote": "Use a queue.",
    }
    monkeypatch.setattr(revision_ui, "create_roadmap",
                        lambda *args, **kwargs: (True, {**data, "answer": "Roadmap", "items": [item]}))
    ui = AppTest.from_file(str(Path(api_client.__file__).with_name("app.py"))).run()
    ui.chat_input[0].set_value("What is a queue?").run()
    assert not ui.exception and any("trích xuất" in item.value for item in ui.warning)
    assert any("[2] graphs.pdf" in item.value for item in ui.markdown)
    navigation = ui.sidebar.radio[0]
    navigation.set_value(navigation.options[2]).run()
    assert not any(widget.label == "Độ khó" for widget in ui.selectbox)
    ui.text_input[0].set_value("tree; graph")
    next(button for button in ui.button if button.label == "Tạo bài trắc nghiệm").click().run()
    assert any("chỉ nhận một chủ đề" in warning.value for warning in ui.warning)
    ui.text_input[0].set_value("trees")
    next(button for button in ui.button if button.label == "Tạo bài trắc nghiệm").click().run()
    assert not any(QUIZ["explanation"] in item.value for item in ui.markdown)
    next(radio for radio in ui.radio if radio.label == "1. What is compared?").set_value(0)
    next(button for button in ui.button if button.label == "Chấm bài").click().run()
    assert not ui.exception and any("1/1" in item.value for item in ui.success)
    navigation = ui.sidebar.radio[0]
    navigation.set_value(navigation.options[3]).run()
    button = next(button for button in ui.button if button.label == "Tạo lộ trình ôn tập")
    button.click().run()
    assert any("Nhập chủ đề" in message.value for message in ui.warning)
    next(field for field in ui.text_input if field.label == "Chủ đề muốn ôn tập").set_value("Graph")
    next(button for button in ui.button if button.label == "Tạo lộ trình ôn tập").click().run()
    assert not ui.exception and not ui.number_input and not ui.date_input
    assert ui.subheader[0].value == "Bước 1 — Graphs"
    assert not ui.checkbox and not any("graphs.pdf" in widget.value for widget in ui.markdown)
