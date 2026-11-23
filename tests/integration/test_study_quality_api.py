"""Exercise uploads, real scoped retrieval and the Study APIs with a mock gateway."""
import json
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import fitz
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from qdrant_client import QdrantClient

from app.api.routes import upload
from app.generation import quiz_generator, rag_engine
from app.main import app
from app.retrieval import indexing
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine

BFS = "Breadth-first search uses a queue to visit vertices in increasing distance from the start vertex."
SORT = "Merge sort divides an array into halves, sorts the halves recursively, and then merges them in order."
PRIVATE = "A private worksheet explains that a directed graph uses ordered pairs for its edges."
BENCH = "The benchmark corpus contains a reference chapter about depth-first search on a directed graph."


def pdf(sentence):
    with fitz.open() as document:
        document.new_page().insert_textbox(
            fitz.Rect(40, 40, 550, 160),
            sentence + " Students can trace the data structure state after each operation to verify the result.",
        )
        return document.tobytes()


def _passages(prompt):
    return {int(label): passage.strip()
            for label, passage in re.findall(r"(?m)^\[(\d+)\][^\n]*\n([^\n]+)", prompt)}


def _revision_tasks(system, prompt):
    passages = _passages(prompt)
    decoder = json.JSONDecoder()
    topics, _ = decoder.raw_decode(system.split("Topics: ", 1)[1])
    assignments, _ = decoder.raw_decode(system.split("Assignments: ", 1)[1])
    sources_by_topic = {topic["id"]: topic["sources"][0] for topic in topics}
    contents = []
    for entry in assignments:
        labels = [sources_by_topic[topic_id] for topic_id in entry["topic_ids"]]
        index = entry["slot"]
        contents.append({
            "slot": index, "objective": f"Objective {index}", "knowledge": ["Study source material"],
            "activity": f"Practice {index}", "practice": "Solve an example",
            "sources": labels, "support_quote": passages[labels[0]],
        })
    return json.dumps(contents)


@pytest.fixture
def study_client(monkeypatch, tmp_path):
    client = QdrantClient(":memory:")
    config = RetrievalConfig(qdrant_mode="memory")
    engine = RetrievalEngine(
        client=client, config=config, query_embedder=lambda _: [1., 0.],
        rerank_scorer=lambda pairs: [.85] * len(pairs),
    )
    monkeypatch.setattr(app.state, "retrieval_engine", engine, raising=False)
    monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(indexing, "embedding_dimension", lambda *args, **kwargs: 2)
    monkeypatch.setattr(indexing, "embed_texts", lambda texts, *args, **kwargs: [[1., 0.]] * len(texts))
    monkeypatch.setattr(rag_engine, "settings", SimpleNamespace(
        LLM_BASE_URL="https://fit.invalid/v1", LLM_MODEL="mock-qwen-27b", OPENAI_API_KEY=SecretStr("fake"),
    ))
    indexing.build_index([{
        "chunk_id": "benchmark", "document_id": "bench", "filename": "bench.pdf", "page": 1, "text": BENCH,
    }], client=client, config=config, workspace_id="research", corpus_id="benchmark-v1")
    try:
        yield TestClient(app)
    finally:
        client.close()


def _upload(client, name, content, workspace):
    response = client.post("/api/upload", files={"file": (name, pdf(content))},
                           data={"workspace_id": workspace})
    assert response.status_code == 200, response.text


def test_upload_retrieve_quiz_and_answer_key(study_client, monkeypatch):
    _upload(study_client, "BFS.pdf", BFS, "alice")
    _upload(study_client, "Sort.pdf", SORT, "alice")
    _upload(study_client, "Private.pdf", PRIVATE, "bob")
    calls = []

    def mock_fit(system, prompt, *, provider, model, max_tokens, task, raise_errors, temperature=None):
        assert temperature in (0.0, 0.7)
        assert provider == "openai" and model == "mock-qwen-27b"
        assert "Private.pdf" not in prompt and "bench.pdf" not in prompt
        passages = _passages(prompt)
        # The topic gate keeps the unrelated Sort.pdf chunk out of a BFS quiz.
        assert len(passages) == 1
        source = next(label for label, passage in passages.items() if BFS in passage)
        calls.append(system)
        if "Audit every question" in system:
            return json.dumps([{"slot": 1, "supported": True, "distinct": True}])
        if "Independently solve" in system:
            return json.dumps([{
                "slot": 1, "correct_index": 1, "unique_answer": True,
                "on_topic": True, "source": source, "quote": BFS,
            }])
        if "Slots: [1," not in system:
            return "[]"
        return json.dumps([{
            "slot": 1, "question": "Which data structure does BFS use?", "options": ["Stack", "Queue", "Heap", "Tree"],
            "correct_index": 1, "explanation": "A queue visits vertices in increasing distance.",
            "sources": [source],
        }])

    monkeypatch.setattr(rag_engine, "generate_answer", mock_fit)
    monkeypatch.setattr(quiz_generator, "MAX_PASSES", 2)
    response = study_client.post("/api/quiz", json={
        "query": "BFS graph traversal", "workspace_id": "alice", "num_questions": 5, "difficulty": "Easy",
    })
    assert response.status_code == 200, response.text
    quiz = response.json()
    assert (quiz["generated_count"], quiz["requested_count"]) == (1, 5)
    assert quiz["generation_error"] == "invalid_output" and "1/5" in quiz["answer"]
    assert len(calls) == 4 and [cite["filename"] for cite in quiz["citations"]] == ["BFS.pdf"]
    assert quiz["questions"][0]["options"][quiz["questions"][0]["correct_index"]] == "Queue"
    assert "difficulty" not in quiz["questions"][0]

    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: pytest.fail("called model"))
    invalid = study_client.post("/api/quiz", json={"topic": "BFS, graph", "workspace_id": "alice"})
    assert invalid.status_code == 422 and "chỉ nhận một chủ đề" in invalid.json()["detail"][0]["msg"]
    missing = study_client.post("/api/quiz", json={"query": "Trie", "workspace_id": "alice"}).json()
    assert missing["generation_mode"] == "abstained" and not missing["questions"]
    assert "không có trong các tài liệu" in missing["answer"]


def test_quiz_one_short_topic_with_grounded_evidence(study_client, monkeypatch):
    _upload(study_client, "BFS.pdf", BFS, "alice")
    _upload(study_client, "Sort.pdf", SORT, "alice")
    _upload(study_client, "Other.pdf", SORT, "bob")

    def mock_fit(system, prompt, **kwargs):
        assert "BFS.pdf" not in prompt and "Other.pdf" not in prompt and "bench.pdf" not in prompt
        passages = _passages(prompt)
        assert len(passages) == 1
        source = next(label for label, text in passages.items() if SORT in text)
        if "Independently solve" in system:
            return json.dumps([{"slot": 1, "correct_index": 0, "unique_answer": True, "on_topic": True,
                                "source": source, "quote": SORT}])
        if "Audit every question" in system:
            return json.dumps([{"slot": 1, "supported": True, "distinct": True}])
        return json.dumps([{
            "slot": 1, "question": "What does merge sort do after sorting both halves?",
            "options": ["Merge them", "Discard one", "Reverse both", "Hash them"],
            "correct_index": 0, "explanation": "It merges the sorted halves in order.", "sources": [source],
        }])

    monkeypatch.setattr(rag_engine, "generate_answer", mock_fit)
    response = study_client.post("/api/quiz", json={"topic": "sort", "num_questions": 1, "workspace_id": "alice"})
    assert response.status_code == 200, response.text
    quiz = response.json()
    assert quiz["requested_count"] == quiz["generated_count"] == 1
    assert quiz["generation_error"] is None and [cite["filename"] for cite in quiz["citations"]] == ["Sort.pdf"]


def test_revision_task_prefills_clean_topic_and_generates_quiz(study_client, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _upload(study_client, "Graph.pdf", BFS, "alice")
    frontend = Path(__file__).resolve().parents[2] / "frontend"
    monkeypatch.syspath_prepend(str(frontend))
    import api_client

    calls = []

    def post(url, *, json, timeout):
        endpoint = urlsplit(url).path
        response = study_client.post(endpoint, json=json)
        calls.append((endpoint, json, response.json()))
        return SimpleNamespace(ok=response.is_success, status_code=response.status_code,
                               json=response.json, headers=response.headers)

    monkeypatch.setattr(api_client, "check_health", lambda: (True, {}))
    monkeypatch.setattr(api_client.requests, "post", post)

    def mock_fit(system, prompt, *, task, **kwargs):
        if task == "revision":
            return _revision_tasks(system, prompt)
        sources = _passages(prompt)
        label = next(label for label, passage in sources.items() if BFS in passage)
        if "Independently solve" in system:
            return json.dumps([{"slot": 1, "correct_index": 0, "unique_answer": True, "on_topic": True,
                                "source": label, "quote": BFS}])
        if "Audit every question" in system:
            return json.dumps([{"slot": 1, "supported": True, "distinct": True}])
        return json.dumps([{
            "slot": 1, "question": "What does BFS use to visit vertices in order?",
            "options": ["Queue", "Stack", "Heap", "Tree"],
            "correct_index": 0, "explanation": "BFS uses a queue.", "sources": [label],
        }])

    monkeypatch.setattr(rag_engine, "generate_answer", mock_fit)
    ui = AppTest.from_file(str(frontend / "app.py")).run()
    ui.sidebar.text_input[0].set_value("alice")
    navigation = ui.sidebar.radio[0]
    navigation.set_value(navigation.options[3]).run()
    ui.checkbox[0].check().run()
    next(button for button in ui.button if button.label == "Tạo kế hoạch học").click().run()
    assert not ui.exception
    assert calls[0][0] == "/api/revision"
    assert calls[0][2]["plan"]["topics"][0]["title"] == "BFS · Graph.pdf"

    next(button for button in ui.button if button.label == "Luyện quiz cho mục này").click().run()
    assert not ui.exception
    assert next(field for field in ui.text_input if field.label == "Chủ đề").value == "BFS"
    next(field for field in ui.number_input if field.label.startswith("Số câu hỏi")).set_value(1)
    next(button for button in ui.button if button.label == "Tạo bài trắc nghiệm").click().run()
    assert not ui.exception
    endpoint, body, quiz = calls[-1]
    assert endpoint == "/api/quiz"
    assert body == {"topic": "BFS", "num_questions": 1, "workspace_id": "alice", "corpus_id": "user-uploads"}
    assert quiz["generated_count"] == 1 and quiz["generation_error"] is None
    assert quiz["citations"][0]["filename"] == "Graph.pdf"


@pytest.mark.parametrize("extra_chapters", [0, 10])
def test_all_topics_scheduler_uses_uploaded_chapters_and_empty_scope(study_client, monkeypatch, extra_chapters):
    _upload(study_client, "Chapter 1 BFS.pdf", BFS, "alice")
    _upload(study_client, "Chapter 2 Sorting.pdf", SORT, "alice")
    for chapter in range(3, extra_chapters + 3):
        _upload(study_client, f"Chapter {chapter} Arrays.pdf",
                "An array stores elements in contiguous memory so an index identifies the position to read.",
                "alice")
    _upload(study_client, "Private.pdf", PRIVATE, "bob")
    calls = []

    def mock_fit(system, prompt, **kwargs):
        assert "Private.pdf" not in prompt and "bench.pdf" not in prompt
        calls.append(system)
        assert kwargs["max_tokens"] <= 4000 and kwargs["task"] == "revision"
        return _revision_tasks(system, prompt)

    monkeypatch.setattr(rag_engine, "generate_answer", mock_fit)
    payload = {"all_topics": True, "days_remaining": 12 if extra_chapters else 5,
               "minutes_per_day": 50 if extra_chapters else 30, "level": "Beginner"}
    response = study_client.post("/api/revision", json={**payload, "workspace_id": "alice"})
    assert response.status_code == 200, response.text
    result = response.json()
    filenames = {item["filename"] for item in result["citations"]}
    assert {"Chapter 1 BFS.pdf", "Chapter 2 Sorting.pdf"} <= filenames
    assert len(filenames) == len(result["plan"]["topics"]) == 2 + extra_chapters
    assert all(sum(task["minutes"] for task in day["tasks"]) <= payload["minutes_per_day"]
               for day in result["plan"]["days"])
    assert result["plan"]["days"][-1]["tasks"][0]["kind"] == "mock_test"
    generated_batches = len(calls)
    assert generated_batches >= (5 if extra_chapters else 2)
    for path, body in (("revision", payload), ("quiz", {"query": "BFS"})):
        empty = study_client.post(f"/api/{path}", json={**body, "workspace_id": "empty"})
        assert empty.status_code == 200 and empty.json()["generation_mode"] == "abstained"
    assert len(calls) == generated_batches
