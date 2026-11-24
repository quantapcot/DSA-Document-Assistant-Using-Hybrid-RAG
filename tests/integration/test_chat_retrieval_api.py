"""Chat API regressions with a scoped retrieval mock and a Grounded Groq mock."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from app.generation import rag_engine
from app.main import app
from app.orchestration.study_sessions import SessionRegistry

GRAPH = "Đồthị G bao gồm một tập các đỉnh và một tập cạnh nối các đỉnh."
BFS = "BFS uses a queue to visit vertices layer by layer from the starting vertex."
DFS = "DFS uses a stack and follows each branch before backtracking to another vertex."
QUICK = "QuickSort divides an array by partitioning against a pivot. Its average running time is O(n log n)."
MERGE = "MergeSort divides an array into halves, sorts each half and merges the result in O(n log n) time."
IRRELEVANT = "Arrays store a series of records in consecutive locations of a memory block."


def hit(section, text, score=.91, **metadata):
    return {"chunk_id": section, "filename": f"{section}.pdf", "section": section, "page": 1,
            "text": text, "score": score, **metadata}


@pytest.fixture
def api(monkeypatch, tmp_path):
    registry = SessionRegistry(tmp_path, 300)
    engine = SimpleNamespace(retrieve=Mock())
    monkeypatch.setitem(app.state._state, "study_sessions", registry)
    monkeypatch.setitem(app.state._state, "retrieval_engine", engine)
    return TestClient(app), engine.retrieve, registry


@pytest.mark.parametrize(("question", "expected", "calls"), [
    ("BFS", {"bfs.pdf"}, 1),
    ("Graph", {"graph.pdf", "bfs.pdf", "dfs.pdf"}, 2),
    ("Đồ thị", {"graph.pdf", "bfs.pdf", "dfs.pdf"}, 2),
    ("Đồ thị là gì?", {"graph.pdf"}, 1),
    ("Các phương pháp duyệt đồ thị", {"bfs.pdf", "dfs.pdf"}, 2),
    ("Các phương pháp Đồ thị", {"graph.pdf", "bfs.pdf", "dfs.pdf"}, 2),
    ("So sánh Quick Sort và Merge Sort", {"quick.pdf", "merge.pdf"}, 2),
    ("Độ phức tạp của Merge Sort", {"merge.pdf"}, 1),
])
def test_chat_covers_the_requested_subjects_before_generation(api, monkeypatch, question, expected, calls):
    client, retrieve, registry = api
    session = registry.create()
    headers = {"Authorization": f"Bearer {session['token']}"}

    def search(*, query, top_k, corpus_id, workspace_id, method):
        assert top_k == 5 and method == "hybrid_rerank"
        assert (corpus_id, workspace_id) == (session["corpus_id"], session["workspace_id"])
        if query == "BFS":
            return [hit("bfs", BFS), hit("noise", IRRELEVANT)]
        if query == "DFS":
            return [hit("dfs", DFS)]
        if "graph traversal" in query:
            return [hit("bfs", BFS), *([hit("dfs", DFS)] if calls == 2 and "duyệt đồ thị" not in question else [])]
        if "đồthị" in query:
            return [hit("noise", IRRELEVANT, .98), hit("graph", GRAPH, .01)]
        if query == "Quick Sort":
            return [hit("quick", QUICK), hit("quick", QUICK), hit("noise", IRRELEVANT)]
        if "Merge Sort" in query:
            return [hit("quick", QUICK, .99), hit("merge", MERGE)]
        raise AssertionError(query)

    retrieve.side_effect = search

    def groq(system, prompt, **kwargs):
        assert all(filename in prompt for filename in expected) and "noise.pdf" not in prompt
        labels = json.loads(prompt.split("Nhãn nguồn theo đối tượng đã xác định: ", 1)[1].split("\n", 1)[0])
        if question in {"Graph", "Đồ thị", "Đồ thị là gì?"}:
            assert "Không chuyển thuộc tính" in system and "ngay sau từng định nghĩa" in system
        if "So sánh" in question:
            assert set(labels) == {"Quick Sort", "Merge Sort"}
            assert labels["Quick Sort"] != labels["Merge Sort"]
            assert "từng đối tượng" in system
        numbers = sorted({label for group in labels.values() for label in group})
        return "Các nguồn liên quan " + " ".join(f"[{label}]" for label in numbers)

    monkeypatch.setattr(rag_engine, "generate_answer", groq)
    response = client.post("/api/chat", json={"query": question}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["generation_mode"] == "llm"
    assert {source["filename"] for source in data["citations"]} == expected
    assert len(data["citations"]) == len(expected)
    assert retrieve.call_count == calls


def test_comparison_is_partial_when_only_one_subject_is_supported(api, monkeypatch):
    client, retrieve, _ = api
    retrieve.side_effect = [[hit("quick", QUICK)], [hit("quick", QUICK)]]
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: rag_engine.NO_EVIDENCE_MESSAGE)
    data = client.post("/api/chat", json={"query": "So sánh Quick Sort và Merge Sort"}).json()
    assert data["generation_mode"] == "extractive"
    assert [source["filename"] for source in data["citations"]] == ["quick.pdf"]
    assert "Chưa tìm được đủ bằng chứng trong lượt truy xuất này cho Merge Sort" in data["answer"]
    assert retrieve.call_count == 2


def test_comparison_drops_model_answer_when_citations_cover_only_one_side(api, monkeypatch):
    client, retrieve, _ = api
    retrieve.side_effect = [[hit("quick", QUICK)], [hit("merge", MERGE)]]
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: "So sánh không đủ nguồn [1].")
    data = client.post("/api/chat", json={"query": "Quick Sort vs Merge Sort"}).json()
    assert data["generation_mode"] == "extractive"
    assert {source["filename"] for source in data["citations"]} == {"quick.pdf", "merge.pdf"}


def test_comparison_uses_separate_sources_when_both_rankings_share_first_hit(api, monkeypatch):
    client, retrieve, _ = api
    shared = hit("shared", "Quick Sort and Merge Sort both divide the input into smaller problems.")
    retrieve.side_effect = [[shared, hit("quick", QUICK)], [shared, hit("merge", MERGE)]]
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: "Các thuật toán chia để trị [1].")
    body = client.post("/api/chat", json={"query": "So sánh Quick Sort và Merge Sort"}).json()
    assert body["generation_mode"] == "extractive"
    assert len(body["citations"]) == 2
    assert "Quick Sort" in body["answer"] and "MergeSort divides" in body["answer"]
    assert {source["filename"] for source in body["citations"]} == {"shared.pdf", "merge.pdf"}


def test_chat_abstains_for_missing_topic_and_empty_workspace(api, monkeypatch):
    client, retrieve, _ = api
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: pytest.fail("LLM called without evidence"))
    retrieve.return_value = [hit("noise", IRRELEVANT)]
    missing = client.post("/api/chat", json={"query": "Suffix automaton"}).json()
    assert missing["generation_mode"] == "abstained" and not missing["citations"]
    assert "chưa đề cập trực tiếp" in missing["answer"]
    retrieve.return_value = []
    monkeypatch.setattr("app.orchestration.request_flow._scope_is_empty", lambda *args: True)
    empty = client.post("/api/chat", json={"query": "BFS"}).json()
    assert "Workspace chưa có tài liệu" in empty["answer"] and not empty["citations"]


def test_chat_keeps_session_scope_even_when_a_bad_result_is_returned(api, monkeypatch):
    client, retrieve, registry = api
    a, b = registry.create(), registry.create()

    def search(*, workspace_id, corpus_id, **kwargs):
        assert corpus_id == a["corpus_id"] == b["corpus_id"]
        own, other = (a, b) if workspace_id == a["workspace_id"] else (b, a)
        return [hit("own", BFS, workspace_id=own["workspace_id"], corpus_id=corpus_id),
                hit("private", BFS, workspace_id=other["workspace_id"], corpus_id=corpus_id)]

    retrieve.side_effect = search

    def groq(system, prompt, **kwargs):
        assert "own.pdf" in prompt and "private.pdf" not in prompt
        return "BFS dùng hàng đợi [1]."

    monkeypatch.setattr(rag_engine, "generate_answer", groq)
    for session in (a, b):
        data = client.post("/api/chat", json={"query": "BFS"},
                           headers={"Authorization": f"Bearer {session['token']}"}).json()
        assert {source["filename"] for source in data["citations"]} == {"own.pdf"}
