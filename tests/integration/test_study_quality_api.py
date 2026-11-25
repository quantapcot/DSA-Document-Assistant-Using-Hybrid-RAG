"""Exercise uploads, real scoped retrieval and the Study APIs with a mock gateway."""
import json
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import fitz
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.api.routes import upload
from app.generation import quiz_generator, rag_engine
from app.main import app
from app.orchestration.study_sessions import SessionRegistry
from app.retrieval import indexing
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine

BFS = "Breadth-first search uses a queue to visit vertices in increasing distance from the start vertex."
SORT = "Merge sort divides an array into halves, sorts the halves recursively, and then merges them in order."
PRIVATE = "A private worksheet explains that a directed graph uses ordered pairs for its edges."
BENCH = "The benchmark corpus contains a reference chapter about depth-first search on a directed graph."
TREE = "A tree is a hierarchical structure of nodes with a root and edges that connect parents and children."
BINARY_TREE = "A binary tree is a tree where each node has at most two children called the left and right child."


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


def _roadmap_items(system, prompt):
    if "Steps to audit: " in system:
        steps, _ = json.JSONDecoder().raw_decode(system.split("Steps to audit: ", 1)[1])
        return json.dumps({"items": [
            {"topic_id": step["topic_id"], "on_topic": True, "supported": True} for step in steps
        ]})
    passages = _passages(prompt)
    topics, _ = json.JSONDecoder().raw_decode(system.split("Assigned topics: ", 1)[1])
    return json.dumps({"items": [{
        "topic_id": topic["id"], "objective": f"Hiểu {topic['title']}",
        "key_knowledge": [f"Ôn {topic['title']}"],
        "sources": [topic["sources"][0]], "support_quote": passages[topic["sources"][0]],
    } for topic in topics]})


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
    monkeypatch.setattr(app.state, "study_sessions", SessionRegistry(tmp_path / "runtime", 3600), raising=False)
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

    def mock_groq(system, prompt, *, response_format, temperature, **kwargs):
        assert temperature in (0.0, 0.7)
        assert response_format["type"] == "json_schema"
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
                "on_topic": True, "self_contained": True, "vietnamese": True, "source": source, "quote": BFS,
            }])
        if "Slots: [1," not in system:
            return "[]"
        return json.dumps([{
            "slot": 1, "question": "Which data structure does BFS use?", "options": ["Stack", "Queue", "Heap", "Tree"],
            "correct_index": 1, "explanation": "A queue visits vertices in increasing distance.",
            "sources": [source],
        }])

    monkeypatch.setattr(rag_engine, "generate_answer", mock_groq)
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

    def mock_groq(system, prompt, **kwargs):
        assert "BFS.pdf" not in prompt and "Other.pdf" not in prompt and "bench.pdf" not in prompt
        passages = _passages(prompt)
        assert len(passages) == 1
        source = next(label for label, text in passages.items() if SORT in text)
        if "Independently solve" in system:
            return json.dumps([{"slot": 1, "correct_index": 0, "unique_answer": True, "on_topic": True, "self_contained": True, "vietnamese": True,
                                "source": source, "quote": SORT}])
        if "Audit every question" in system:
            return json.dumps([{"slot": 1, "supported": True, "distinct": True}])
        return json.dumps([{
            "slot": 1, "question": "What does merge sort do after sorting both halves?",
            "options": ["Merge them", "Discard one", "Reverse both", "Hash them"],
            "correct_index": 0, "explanation": "It merges the sorted halves in order.", "sources": [source],
        }])

    monkeypatch.setattr(rag_engine, "generate_answer", mock_groq)
    response = study_client.post("/api/quiz", json={"topic": "sort", "num_questions": 1, "workspace_id": "alice"})
    assert response.status_code == 200, response.text
    quiz = response.json()
    assert quiz["requested_count"] == quiz["generated_count"] == 1
    assert quiz["generation_error"] is None and [cite["filename"] for cite in quiz["citations"]] == ["Sort.pdf"]


def test_session_roadmap_shows_only_requested_learning_steps(study_client, monkeypatch):
    from streamlit.testing.v1 import AppTest

    frontend = Path(__file__).resolve().parents[2] / "frontend"
    monkeypatch.syspath_prepend(str(frontend))
    import api_client

    calls = []

    def post(url, *, timeout, json=None, headers=None):
        endpoint = urlsplit(url).path
        response = study_client.post(endpoint, json=json, headers=headers)
        calls.append((endpoint, json, response.json()))
        return SimpleNamespace(ok=response.is_success, status_code=response.status_code,
                               json=response.json, headers=response.headers)

    monkeypatch.setattr(api_client, "check_health", lambda: (True, {}))
    monkeypatch.setattr(api_client.requests, "post", post)

    def mock_groq(system, prompt, *, task, **kwargs):
        assert task == "revision"
        assert "Sorting.pdf" not in prompt and "PrivateTree.pdf" not in prompt and "bench.pdf" not in prompt
        return _roadmap_items(system, prompt)

    monkeypatch.setattr(rag_engine, "generate_answer", mock_groq)
    ui = AppTest.from_file(str(frontend / "app.py")).run()
    session = calls[0][2]
    assert session["corpus_id"] == "study-sessions"
    assert not ui.sidebar.text_input
    headers = {"Authorization": f"Bearer {session['token']}"}
    for filename, content in [("A-Binary.pdf", BINARY_TREE), ("Z-Tree.pdf", TREE), ("Sorting.pdf", SORT)]:
        response = study_client.post("/api/upload", headers=headers, files={"file": (filename, pdf(content))})
        assert response.status_code == 200
    other = study_client.post("/api/sessions").json()
    other_headers = {"Authorization": f"Bearer {other['token']}"}
    assert study_client.post("/api/upload", headers=other_headers,
                             files={"file": ("PrivateTree.pdf", pdf(TREE))}).status_code == 200
    navigation = ui.sidebar.radio[0]
    navigation.set_value(navigation.options[3]).run()
    next(field for field in ui.text_input if field.label == "Chủ đề muốn ôn tập").set_value("Tree")
    next(button for button in ui.button if button.label == "Tạo lộ trình ôn tập").click().run()
    assert not ui.exception
    roadmap = next(item for item in calls if item[0] == "/api/roadmap")
    assert roadmap[1] == {"topic": "Tree"}
    items = roadmap[2]["items"]
    assert [item["title"] for item in items] == ["Tree Fundamentals", "Binary Tree"]
    assert all(item["objective"] and item["key_knowledge"] and "activity" not in item for item in items)
    assert {source["filename"] for source in roadmap[2]["citations"]} == {"A-Binary.pdf", "Z-Tree.pdf"}
    assert [item.value for item in ui.subheader] == ["Bước 1 — Tree Fundamentals", "Bước 2 — Binary Tree"]
    assert any("Kiến thức trọng tâm" in item.value for item in ui.markdown)
    assert all(".pdf" not in item.value for item in ui.markdown) and not ui.get("popover")
    assert not ui.checkbox and not ui.number_input and not ui.date_input
    assert not any("Luyện quiz" in button.label for button in ui.button)

    next(button for button in ui.sidebar.button if button.label == "Tạo phiên học mới").click().run()
    fresh = calls[-1][2]
    assert fresh["workspace_id"] != session["workspace_id"]
    assert not ui.exception and not ui.chat_message
    fresh_auth = {"Authorization": f"Bearer {fresh['token']}"}
    assert study_client.get("/api/documents", headers=fresh_auth).json()["total"] == 0
    navigation = ui.sidebar.radio[0]
    navigation.set_value(navigation.options[3]).run()
    assert not ui.subheader


@pytest.mark.parametrize("extra_chapters", [0, 10])
def test_roadmap_excludes_unrelated_chapters_and_distinguishes_empty_scope(study_client, monkeypatch, extra_chapters):
    _upload(study_client, "Chapter 1 BFS.pdf", BFS, "alice")
    _upload(study_client, "Chapter 2 Sorting.pdf", SORT, "alice")
    for chapter in range(3, extra_chapters + 3):
        _upload(study_client, f"Chapter {chapter} Arrays.pdf",
                "An array stores elements in contiguous memory so an index identifies the position to read.",
                "alice")
    _upload(study_client, "Private.pdf", PRIVATE, "bob")
    calls = []

    def mock_groq(system, prompt, **kwargs):
        assert "Private.pdf" not in prompt and "bench.pdf" not in prompt and "Sorting.pdf" not in prompt
        calls.append(system)
        assert kwargs["max_tokens"] <= 4000 and kwargs["task"] == "revision"
        return _roadmap_items(system, prompt)

    monkeypatch.setattr(rag_engine, "generate_answer", mock_groq)
    response = study_client.post("/api/roadmap", json={"workspace_id": "alice", "topic": "Graph"})
    assert response.status_code == 200, response.text
    result = response.json()
    filenames = {item["filename"] for item in result["citations"]}
    assert filenames == {"Chapter 1 BFS.pdf"}
    assert len(result["items"]) == 1
    generated_batches = len(calls)
    assert generated_batches == 2
    for path, body in (("roadmap", {"topic": "Graph"}), ("quiz", {"query": "BFS"})):
        empty = study_client.post(f"/api/{path}", json={**body, "workspace_id": "empty"})
        assert empty.status_code == 200 and empty.json()["generation_mode"] == "abstained"
        assert "chưa có tài liệu" in empty.json()["answer"]
    irrelevant = study_client.post("/api/roadmap", json={"workspace_id": "alice", "topic": "Disjoint Set"})
    assert irrelevant.status_code == 200 and irrelevant.json()["generation_mode"] == "abstained"
    assert "Disjoint Set" in irrelevant.json()["answer"]
    assert study_client.post("/api/roadmap", json={"workspace_id": "alice", "topic": " "}).status_code == 422
    assert len(calls) == generated_batches
