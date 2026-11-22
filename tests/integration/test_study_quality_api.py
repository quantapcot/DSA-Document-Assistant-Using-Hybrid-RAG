"""Exercise uploads, real scoped retrieval and the Study APIs with a mock gateway."""
from collections import defaultdict
from datetime import date, timedelta
import json
import re
from types import SimpleNamespace

import fitz
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from qdrant_client import QdrantClient

from app.api.routes import upload
from app.generation import rag_engine
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

    def mock_fit(system, prompt, *, provider, model, max_tokens):
        assert provider == "openai" and model == "mock-qwen-27b"
        assert "Private.pdf" not in prompt and "bench.pdf" not in prompt
        passages = _passages(prompt)
        assert len(passages) == 2
        source = next(label for label, passage in passages.items() if BFS in passage)
        calls.append(system)
        if "Independently solve" in system:
            return json.dumps([{
                "slot": 1, "correct_index": 1, "unique_answer": True,
                "explanation_supported": True, "source": source, "quote": BFS,
            }])
        return json.dumps([{
            "slot": 1, "question": "Which data structure does BFS use?", "options": ["Stack", "Queue", "Heap", "Tree"],
            "correct_index": 1, "explanation": "A queue visits vertices in increasing distance.",
            "sources": [source], "question_type": "Concept", "difficulty": "Easy", "topic": "BFS",
        }])

    monkeypatch.setattr(rag_engine, "generate_answer", mock_fit)
    response = study_client.post("/api/quiz", json={
        "query": "BFS graph traversal", "workspace_id": "alice", "num_questions": 5, "difficulty": "Easy",
    })
    assert response.status_code == 200, response.text
    quiz = response.json()
    assert (quiz["generated_count"], quiz["requested_count"]) == (1, 5)
    assert len(calls) == 2 and [cite["filename"] for cite in quiz["citations"]] == ["BFS.pdf"]
    assert quiz["questions"][0]["options"][quiz["questions"][0]["correct_index"]] == "Queue"


def test_all_topics_scheduler_uses_uploaded_chapters_and_empty_scope(study_client, monkeypatch):
    _upload(study_client, "Chapter 1 BFS.pdf", BFS, "alice")
    _upload(study_client, "Chapter 2 Sorting.pdf", SORT, "alice")
    _upload(study_client, "Private.pdf", PRIVATE, "bob")
    calls = []

    def mock_fit(system, prompt, **kwargs):
        assert "Private.pdf" not in prompt and "bench.pdf" not in prompt
        passages = _passages(prompt)
        assert any(BFS in part for part in passages.values()) and any(SORT in part for part in passages.values())
        calls.append(system)
        decoder = json.JSONDecoder()
        topics, _ = decoder.raw_decode(system.split("Topics: ", 1)[1])
        assignments, _ = decoder.raw_decode(system.split("Assignments in order: ", 1)[1])
        days = defaultdict(list)
        for index, entry in enumerate(assignments, 1):
            label = next(topic["sources"][0] for topic in topics if topic["id"] == entry["topic_id"])
            days[entry["date"]].append({
                **{k: v for k, v in entry.items() if k != "date"},
                "objective": f"Objective {index}", "knowledge": ["Study source material"],
                "activity": f"Practice {index}", "practice": "Solve an example", "prerequisites": [],
                "sources": [label], "support_quote": passages[label],
            })
        return json.dumps({
            "exam_date": (date.fromisoformat(max(days)) + timedelta(days=1)).isoformat(),
            "minutes_per_day": 30, "level": "Beginner",
            "topics": topics, "days": [{"date": day, "tasks": tasks} for day, tasks in days.items()],
        })

    monkeypatch.setattr(rag_engine, "generate_answer", mock_fit)
    payload = {"all_topics": True, "days_remaining": 5, "minutes_per_day": 30, "level": "Beginner"}
    response = study_client.post("/api/revision", json={**payload, "workspace_id": "alice"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert {item["filename"] for item in result["citations"]} == {"Chapter 1 BFS.pdf", "Chapter 2 Sorting.pdf"}
    assert len(result["plan"]["topics"]) == 2
    assert all(sum(task["minutes"] for task in day["tasks"]) <= 30 for day in result["plan"]["days"])
    assert result["plan"]["days"][-1]["tasks"][0]["kind"] == "mock_test"
    for path, body in (("revision", payload), ("quiz", {"query": "BFS"})):
        empty = study_client.post(f"/api/{path}", json={**body, "workspace_id": "empty"})
        assert empty.status_code == 200 and empty.json()["generation_mode"] == "abstained"
    assert len(calls) == 1
