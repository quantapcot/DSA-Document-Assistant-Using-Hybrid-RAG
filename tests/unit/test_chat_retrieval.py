"""Focused query plans and evidence admission for Chat."""
import asyncio

import pytest

from app.orchestration.chat_retrieval import retrieve_chat_evidence
from app.orchestration.intent_router import MAX_CHAT_RETRIEVALS, plan_chat_query


@pytest.mark.parametrize(("query", "intent", "names"), [
    ("BFS", "simple", ["BFS"]),
    ("Graph", "overview", ["Graph"]),
    ("Đồ thị", "overview", ["Graph"]),
    ("Đồ thị là gì?", "overview", ["Graph"]),
    ("Các phương pháp duyệt đồ thị", "overview", ["Graph traversal"]),
    ("So sánh Quick Sort và Merge Sort", "comparison", ["Quick Sort", "Merge Sort"]),
    ("AVL và BST khác nhau như thế nào?", "comparison", ["AVL", "Binary Search Tree"]),
    ("Compare AVL, BST and splay tree", "comparison", ["AVL", "Binary Search Tree", "splay tree"]),
    ("Độ phức tạp của Merge Sort", "explanation", ["Merge Sort"]),
    ("How does union-find work?", "explanation", ["union-find"]),
    ("What is a splay tree?", "explanation", ["splay tree"]),
])
def test_chat_query_intents_and_targets(query, intent, names):
    plan = plan_chat_query(query)
    assert plan.intent == intent
    assert [target.name for target in plan.targets] == names


def test_bilingual_graph_plans_search_equivalent_evidence():
    assert plan_chat_query("Graph").targets == plan_chat_query("Đồ thị").targets
    assert "đồthị" in plan_chat_query("Đồ thị").targets[0].query
    assert not plan_chat_query("Đồ thị là gì?").followups


def test_comparison_plan_bounds_searches_and_tracks_remaining_targets():
    plan = plan_chat_query("Compare AVL, BFS, DFS, Merge Sort and red black tree")
    assert len(plan.targets) == MAX_CHAT_RETRIEVALS
    assert plan.unsearched == ("red black tree",)


@pytest.mark.parametrize(("results", "reason"), [
    ([], "no_results"),
    ([("An unrelated array stores consecutive records in contiguous memory.", 0.9)], "off_topic"),
    ([("This unrelated introduction lists several topics in an alphabetical appendix including BFS.", 0.01)], "weak_evidence"),
])
def test_no_context_distinguishes_missing_off_topic_and_weak_results(results, reason):
    async def search(query):
        return [{"filename": "lesson.pdf", "text": text, "score": score} for text, score in results], "hybrid_rerank"

    context = asyncio.run(retrieve_chat_evidence(plan_chat_query("BFS"), search, "session", "corpus"))
    assert not context.passages and context.reason == reason


def test_focused_low_score_definition_has_independent_document_support():
    async def search(query):
        return [{"filename": "lesson.pdf", "text": "Đồthị G bao gồm một tập đỉnh cùng với một tập cạnh.", "score": .02}], "hybrid_rerank"

    context = asyncio.run(retrieve_chat_evidence(plan_chat_query("Đồ thị là gì?"), search, "session", "corpus"))
    assert context.covered == {"Graph": (1,)}
    assert context.reason is None and context.calls == 1


def test_bst_english_and_vietnamese_use_the_same_query():
    assert plan_chat_query("Binary Search Tree").targets == plan_chat_query("Cây nhị phân tìm kiếm").targets
