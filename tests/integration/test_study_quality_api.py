"""Exercise uploads, real scoped retrieval and the Study APIs with a mock gateway."""
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

    def mock_fit(system, prompt, *, provider, model, max_tokens, task, raise_errors):
        assert provider == "openai" and model == "mock-qwen-27b"
        assert "Private.pdf" not in prompt and "bench.pdf" not in prompt
        passages = _passages(prompt)
        assert len(passages) == 2
        source = next(label for label, passage in passages.items() if BFS in passage)
        calls.append(system)
        if "Audit each explanation" in system:
            return json.dumps([{"slot": 1, "supported": True}])
        if "Independently solve" in system:
            return json.dumps([{
                "slot": 1, "correct_index": 1, "unique_answer": True,
                "on_topic": True, "source": source, "quote": BFS,
            }])
        if '"1":' not in system:
            return "[]"
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
    assert len(calls) == 4 and [cite["filename"] for cite in quiz["citations"]] == ["BFS.pdf"]
    assert quiz["questions"][0]["options"][quiz["questions"][0]["correct_index"]] == "Queue"


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
        passages = _passages(prompt)
        calls.append(system)
        assert kwargs["max_tokens"] <= 4000 and kwargs["task"] == "revision"
        decoder = json.JSONDecoder()
        topics, _ = decoder.raw_decode(system.split("Topics: ", 1)[1])
        assignments, _ = decoder.raw_decode(system.split("Assignments: ", 1)[1])
        contents = []
        for entry in assignments:
            labels = [next(topic["sources"][0] for topic in topics if topic["id"] == topic_id)
                      for topic_id in entry["topic_ids"]]
            index = entry["slot"]
            contents.append({
                "slot": index, "objective": f"Objective {index}", "knowledge": ["Study source material"],
                "activity": f"Practice {index}", "practice": "Solve an example",
                "sources": labels, "support_quote": passages[labels[0]],
            })
        return json.dumps(contents)

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
