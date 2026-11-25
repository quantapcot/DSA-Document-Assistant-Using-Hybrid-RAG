import json
import logging

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.generation import llm_client, quiz_generator, rag_engine, revision_generator
from app.generation.llm_client import LLMError
from app.generation.study_output import decode_json, supports_quote
from app.generation.study_planner import (
    build_curriculum, curriculum_dependencies, order_topics,
)
from app.orchestration.study_evidence import diverse_evidence
from app.schemas.contracts import QuizRequest, RoadmapRequest

QUOTE = "Breadth-first search visits vertices in ascending order of distance from the start."


def evidence(number=3):
    return [{
        "chunk_id": str(i), "document_id": str(i), "filename": f"Chapter {i}.pdf", "page": 1,
        "section": f"Section {i}", "text": f"{QUOTE} Example {i}.", "workspace_id": "student",
        "corpus_id": "uploads", "score": .8, "retrieval_method": "hybrid_rerank",
    } for i in range(1, number + 1)]


def question(slot, *, text=None):
    return {
        "slot": slot, "question": text or f"Which invariant determines the BFS traversal for case {slot}?",
        "options": ["distance", "vertex label", "stack size", "edge weight"],
        "correct_index": 0, "explanation": "The order follows distance from the start.",
        "sources": [1],
    }


def verification(slot, **overrides):
    return {
        "slot": slot, "correct_index": 0, "unique_answer": True, "on_topic": True,
        "self_contained": True, "vietnamese": True,
        "source": 1, "quote": QUOTE, **overrides,
    }


def test_diversity_deduplicates_and_drops_wrong_scope():
    a, b, c = evidence()
    hits = diverse_evidence(
        [[{**a, "score": .01}, {**b, "workspace_id": "other"}, c], [a, {**a, "chunk_id": "copy"}, b]],
        workspace="student", corpus="uploads",
    )
    assert [item["chunk_id"] for item in hits] == ["1", "3", "2"]
    assert hits[0]["section"] == "Section 1"


@pytest.mark.parametrize(("overrides", "report", "audit", "reason"), [
    ({"correct_index": 4}, {}, {}, "field=correct_index"),
    ({"options": ["one", "Two", "two!", "three"]}, {}, {}, "non_distinct_options"),
    ({"sources": [99]}, {}, {}, "invalid_citations"),
    ({}, {"source": 2}, {}, "uncited_source"),
    ({}, {"correct_index": 1}, {}, "answer_mismatch"),
    ({}, {"unique_answer": False}, {}, "answer_not_unique"),
    ({}, {"on_topic": False}, {}, "off_topic"),
    ({}, {"self_contained": False}, {}, "missing_question_context"),
    ({}, {"vietnamese": False}, {}, "non_vietnamese_question"),
    ({"question": "According to the provided text, what is the answer?"}, {}, {}, "hidden_source_reference"),
    ({"question": "Dựa trên tài liệu, BFS dùng cấu trúc nào?"}, {}, {}, "hidden_source_reference"),
    ({"question": "As shown above, which vertex comes first?"}, {}, {}, "hidden_source_reference"),
    ({}, {"quote": "A fabricated sentence that is absent from the file."}, {}, "unsupported_excerpt"),
    ({}, {}, {"supported": False}, "unsupported_explanation"),
    ({}, {}, {"distinct": False}, "repeated_question"),
])
def test_rejected_questions_never_fill_a_missing_slot(monkeypatch, caplog, overrides, report, audit, reason):
    def mock(query, passages, instructions, *, stage, **kwargs):
        if stage == "quiz.candidate":
            return [{**question(1), **overrides}]
        if stage == "quiz.checker":
            return [verification(1, **report)]
        return [{"slot": 1, "supported": True, "distinct": True, **audit}]

    monkeypatch.setattr(quiz_generator, "generate_study_json", mock)
    monkeypatch.setattr(quiz_generator, "MAX_PASSES", 2)
    result = quiz_generator.generate_quiz("BFS", evidence(), num_questions=1)
    assert not result.questions and not result.citations
    assert (result.generated_count, result.requested_count) == (0, 1)
    assert result.generation_error == "invalid_output" and "0/1" in result.answer
    assert reason in caplog.text


def test_invalid_json_and_missing_evidence_are_explicit(monkeypatch):
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: "{broken json")
    invalid = quiz_generator.generate_quiz("BFS", evidence())
    assert invalid.generation_mode == "unavailable" and invalid.generation_error == "invalid_json"
    assert "0/5" in invalid.answer
    result = quiz_generator.generate_quiz("BFS", [], workspace_empty=True)
    assert result.generation_mode == "abstained" and not result.questions and "0/5" in result.answer
    for invalid in (0, 6):
        with pytest.raises(ValidationError):
            QuizRequest(topic="BFS", num_questions=invalid)


@pytest.mark.parametrize("count", [1, 3, 5])
def test_quiz_exact_count_and_refills_only_rejected_slots(monkeypatch, count):
    prompts = [
        "Where are duplicate keys stored?", "Describe the visitation order.",
        "Can the root be removed safely?", "Pick the matching recurrence.", "Compare space utilization.",
    ]
    drafts = []

    def llm(query, passages, instructions, *, stage, **kwargs):
        if stage == "quiz.candidate":
            assert "difficulty" not in instructions and "question_type" not in instructions
            raw = instructions.split("Slots: ", 1)[1]
            assignments, _ = json.JSONDecoder().raw_decode(raw)
            drafts.append(assignments)
            if len(drafts) == 2:
                # A model sometimes restarts slot numbering at 1 on a refill.
                return [question(1, text="What changes when the graph is disconnected?")]
            return [question(slot, text=prompts[slot - 1]) for slot in assignments]
        supplied = json.loads(query)
        if stage == "quiz.checker":
            assert all("explanation" not in item and "correct_index" not in item for item in supplied)
            assert list(passages) == [1]
            return [verification(item["slot"]) for item in supplied]
        assert all("verified_answer" in item and "explanation" in item for item in supplied["questions"])
        if len(drafts) == 2:
            assert len(supplied["previous"]) == count - 1
        return [
            {"slot": item["slot"], "supported": True,
             "distinct": item["slot"] != count or len(drafts) == 2}
            for item in supplied["questions"]
        ]

    monkeypatch.setattr(quiz_generator, "generate_study_json", llm)
    result = quiz_generator.generate_quiz("BFS", evidence(), num_questions=count)
    assert result.requested_count == result.generated_count == count
    assert drafts == [list(range(1, count + 1)), [count]]
    assert result.generation_error is None and result.citations[0].filename == "Chapter 1.pdf"
    assert all(item.topic == "BFS" for item in result.questions)


def test_quiz_refill_timeout_preserves_verified_questions(monkeypatch):
    stages = iter([
        [question(1)], [verification(1)], [{"slot": 1, "supported": True, "distinct": True}],
    ])

    def mock(*args, **kwargs):
        result = next(stages, None)
        if result is None:
            raise LLMError("timeout")
        return result

    monkeypatch.setattr(quiz_generator, "generate_study_json", mock)
    result = quiz_generator.generate_quiz("BFS", evidence(), num_questions=3)
    assert (result.generated_count, result.requested_count) == (1, 3)
    assert result.generation_error == "timeout" and "1/3" in result.answer


def test_exact_duplicates_are_removed_even_if_auditor_approves_them(monkeypatch, caplog):
    def mock(query, passages, instructions, *, stage, **kwargs):
        if stage == "quiz.candidate":
            return [question(1), question(2, text=question(1)["question"])]
        inputs = json.loads(query)
        if stage == "quiz.checker":
            return [verification(item["slot"]) for item in inputs]
        return [{"slot": item["slot"], "supported": True, "distinct": True}
                for item in inputs["questions"]]

    monkeypatch.setattr(quiz_generator, "generate_study_json", mock)
    monkeypatch.setattr(quiz_generator, "MAX_PASSES", 1)
    result = quiz_generator.generate_quiz("BFS", evidence(), num_questions=2)
    assert result.generated_count == 1 and "1/2" in result.answer
    assert "reason=duplicate_question" in caplog.text


def test_quiz_provider_timeout_logs_actual_attempts_without_secrets(monkeypatch, caplog):
    config = Settings(_env_file=None, GROQ_API_KEY="PRIVATE-MARKER")
    monkeypatch.setattr(llm_client, "settings", config)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    attempts = []

    def call(*args, timeout, **kwargs):
        attempts.append(timeout)
        raise TimeoutError("PRIVATE-MARKER")

    monkeypatch.setattr(llm_client, "_call_groq", call)
    with caplog.at_level(logging.INFO):
        result = quiz_generator.generate_quiz("BFS", evidence(), num_questions=1)
    assert attempts == [config.LLM_QUIZ_GENERATION_TIMEOUT] * 3
    assert result.generation_error == "timeout" and not result.questions
    assert "0/1" in result.answer
    assert "llm_requests=1 provider_calls=3" in caplog.text and "elapsed_ms=" in caplog.text
    assert "PRIVATE-MARKER" not in caplog.text


def _roadmap_drafts(instructions, sources):
    catalog, _ = json.JSONDecoder().raw_decode(instructions.split("Assigned topics: ", 1)[1])
    return [{
        "topic_id": topic["id"], "objective": f"Hiểu {topic['title']}",
        "key_knowledge": [f"Nguyên lý của {topic['title']}"],
        "sources": [topic["sources"][0]],
        "support_quote": sources[topic["sources"][0]]["text"],
    } for topic in catalog]


def _roadmap_reviews(instructions):
    steps, _ = json.JSONDecoder().raw_decode(instructions.split("Steps to audit: ", 1)[1])
    return [{"topic_id": step["topic_id"], "on_topic": True, "supported": True} for step in steps]


def test_roadmap_has_no_calendar_inputs_and_abstains_when_empty(monkeypatch):
    with pytest.raises(ValidationError):
        RoadmapRequest(topic="Tree", days_remaining=4)
    for topic in ("", "   "):
        with pytest.raises(ValidationError):
            RoadmapRequest(topic=topic)
    assert RoadmapRequest(topic="  Tree  ").topic == "Tree"
    monkeypatch.setattr(revision_generator, "generate_study_json", lambda *args, **kwargs: pytest.fail("LLM called"))
    empty = revision_generator.generate_roadmap("Tree", [])
    irrelevant = revision_generator.generate_roadmap("Tree", [], has_documents=True)
    assert not empty.items and empty.generation_mode == "abstained" and "chưa có tài liệu" in empty.answer
    assert irrelevant.generation_mode == "abstained" and "Tree" in irrelevant.answer


def test_roadmap_deduplicates_and_orders_prerequisites(monkeypatch):
    bfs, graph, duplicate, alias = evidence(4)
    bfs = {**bfs, "section": "BFS", "text": QUOTE + " BFS requires an understanding of a graph."}
    graph = {**graph, "section": "Graph", "text": "A graph represents vertices and edges between them in a structure."}
    alias = {**alias, "section": "Đồ thị", "text": "A graph is represented with an adjacency list or matrix."}
    duplicate = {**duplicate, "section": "BFS", "text": bfs["text"]}
    documents = [bfs, graph, duplicate, alias]
    catalog, passages = build_curriculum(documents, "Graph")
    assert len(catalog) == 2 and len(passages) == 3
    assert [topic.title for topic in order_topics(catalog, curriculum_dependencies(catalog, passages))] == ["Đồ thị", "BFS"]

    calls = []
    def mock(query, sources, instructions, **kwargs):
        calls.append(kwargs)
        if kwargs["stage"] == "roadmap.review":
            return {"items": _roadmap_reviews(instructions)}
        return {"items": _roadmap_drafts(instructions, sources)}

    monkeypatch.setattr(revision_generator, "generate_study_json", mock)
    result = revision_generator.generate_roadmap("Graph", documents)
    assert [call["row_model"] for call in calls] == [revision_generator.RoadmapDraft, revision_generator.RoadmapReview]
    assert len(result.items) == 2 and result.items[1].prerequisites == [result.items[0].id]
    assert {source.filename for source in result.citations} == {bfs["filename"], graph["filename"]}


@pytest.mark.parametrize("problem", ["unknown_topic", "wrong_source", "fabricated_quote", "invented_date", "duplicate"])
def test_roadmap_rejects_unfounded_learning_content(monkeypatch, problem):
    documents = [{**evidence(1)[0], "section": "BFS"}]

    def mock(query, sources, instructions, **kwargs):
        drafts = _roadmap_drafts(instructions, sources)
        updates = {
            "unknown_topic": {"topic_id": "invented"}, "wrong_source": {"sources": [99]},
            "fabricated_quote": {"support_quote": "A fabricated sentence that is not in the supplied passages."},
            "invented_date": {"date": "invalid"},
        }
        if problem == "duplicate":
            drafts.append(drafts[0].copy())
        else:
            drafts[0].update(updates[problem])
        return {"items": drafts}

    monkeypatch.setattr(revision_generator, "generate_study_json", mock)
    result = revision_generator.generate_roadmap("BFS", documents)
    assert not result.items and result.generation_error == "invalid_output"
    assert result.uncovered_topics == ["BFS"]


def test_roadmap_retries_truncated_json_once(monkeypatch):
    documents = [{**evidence(1)[0], "section": "BFS"}]
    calls = []

    def mock(query, sources, instructions, **kwargs):
        if kwargs["stage"] == "roadmap.review":
            return {"items": _roadmap_reviews(instructions)}
        calls.append(kwargs["temperature"])
        if len(calls) == 1:
            raise LLMError("truncated_json")
        return {"items": _roadmap_drafts(instructions, sources)}

    monkeypatch.setattr(revision_generator, "generate_study_json", mock)
    result = revision_generator.generate_roadmap("BFS", documents)
    assert len(result.items) == 1 and calls == [0.0, 0.2]


def test_roadmap_tree_normalizes_headings_and_orders_from_source_relationships():
    descriptions = [
        ("AVL", "AVL is a binary search tree that keeps the height of its subtrees balanced."),
        ("BST (Binary Search Tree)", "A binary search tree is a binary tree with a key ordering rule."),
        ("2.0 The binary tree is an example covered in this material",
         "A binary tree is a tree where each node has at most two children."),
        ("Tree Fundamentals", "A tree organizes nodes in a hierarchy with a root and parent-child relationships."),
        ("Tree traversal visits nodes in order",
         "Tree traversal requires a binary tree in these examples and visits its nodes in order."),
        ("Sorting", "Insertion sort places every key in the correct position in a sorted prefix."),
    ]
    documents = [{**item, "section": section, "text": text}
                 for item, (section, text) in zip(evidence(len(descriptions)), descriptions)]
    catalog, passages = build_curriculum(documents, "Tree")
    graph = curriculum_dependencies(catalog, passages)
    titles = [entry.title for entry in order_topics(catalog, graph)]
    assert set(titles) == {"Tree Fundamentals", "Binary Tree", "Binary Search Tree", "AVL Tree", "Tree Traversal"}
    assert titles.index("Tree Fundamentals") < titles.index("Binary Tree") < titles.index("Binary Search Tree")
    assert titles.index("AVL Tree") > titles.index("Binary Search Tree")
    assert titles.index("Tree Traversal") > titles.index("Binary Tree")


def test_roadmap_can_discover_free_form_headings_without_accepting_sentences():
    a, b, c = evidence()
    documents = [
        {**a, "section": "Path Compression", "text": "Disjoint sets shorten paths by pointing nodes to the root."},
        {**b, "section": "This section explains why disjoint set operations are useful.",
         "text": "A disjoint set maintains nonoverlapping groups of elements."},
        {**c, "section": "Tree", "text": "Trees organize data hierarchically using a root and children."},
    ]
    catalog, _ = build_curriculum(documents, "Disjoint Set")
    assert [entry.title for entry in catalog] == ["Path Compression", "Disjoint Set"]


def test_roadmap_can_discover_multiple_concepts_from_one_passage():
    [base] = evidence(1)
    item = {**base, "section": None, "text": (
        "A tree organizes nodes around a root and assigns each non-root node a parent. "
        "A binary tree is a tree with at most two children per node."
    )}
    catalog, passages = build_curriculum([item], "Tree")
    assert [entry.title for entry in catalog] == ["Tree Fundamentals", "Binary Tree"]
    assert len(passages) == 1 and all(entry.sources == (1,) for entry in catalog)


@pytest.mark.parametrize(("topic", "text", "title"), [
    ("Sorting", "Insertion sort inserts each item into the proper position in the sorted prefix.", "Insertion Sort"),
    ("Dijkstra", "Dijkstra computes shortest paths in a graph with nonnegative edge weights.", "Dijkstra"),
    ("Graph", "Breadth-first search visits vertices in ascending order of distance from the start.", "BFS"),
])
def test_roadmap_discovers_named_subtopics_from_unstructured_text(topic, text, title):
    passage = {**evidence(1)[0], "section": None, "text": text}
    catalog, _ = build_curriculum([passage], topic)
    assert [entry.title for entry in catalog] == [title]


@pytest.mark.parametrize("broken", ["schema", "unsupported", "off_topic", "invalid_json", "wrong_json_shape", "recovered_row"])
def test_roadmap_retains_valid_steps_when_a_neighbor_fails(monkeypatch, broken):
    a, b = evidence(2)
    passages = [
        {**a, "section": "Tree Fundamentals", "text": "A tree has a root node and edges that organize its children."},
        {**b, "section": "Binary Tree", "text": "A binary tree is a tree where each node has at most two children."},
    ]

    def mock(query, sources, instructions, *, stage, **kwargs):
        if stage == "roadmap.review":
            reports = _roadmap_reviews(instructions)
            for report in reports:
                if report["topic_id"] == "T1":
                    report.update(supported=broken != "unsupported", on_topic=broken != "off_topic")
            return {"items": reports}
        drafts = _roadmap_drafts(instructions, sources)
        if broken in {"invalid_json", "wrong_json_shape", "recovered_row"} and any(row["topic_id"] == "T1" for row in drafts):
            if broken == "wrong_json_shape":
                return {"items": "invalid"}
            if broken == "recovered_row" and len(drafts) == 2:
                return drafts[1]
            raise LLMError("invalid_json")
        for draft in drafts:
            if draft["topic_id"] == "T1":
                if broken == "schema":
                    draft["key_knowledge"] = []
                elif broken in {"unsupported", "off_topic"}:
                    draft["key_knowledge"] = ["Bubble Sort luôn chạy trong O(1)."]
        return {"items": drafts}

    monkeypatch.setattr(revision_generator, "generate_study_json", mock)
    result = revision_generator.generate_roadmap("Tree", passages)
    assert [item.title for item in result.items] == ["Binary Tree"]
    assert result.items[0].prerequisites == []
    assert result.uncovered_topics == ["Tree Fundamentals"]
    assert result.items[0].key_knowledge


def test_quiz_one_accepts_unambiguous_json_representations(monkeypatch):
    draft = {
        **question(1), "slot": "1", "sources": ["[1]"],
    }
    review = verification(
        "1", correct_index="0", source="[1]", unique_answer="true", on_topic="True",
        self_contained="true", vietnamese="True",
        quote="“BREADTH-FIRST SEARCH visits vertices in ascending order of distance from the start!”",
    )
    answers = iter([
        "broken JSON",
        "<think>reasoning</think>\n```json\n" + json.dumps({"questions": [draft]}) + "\n```",
        json.dumps({"checks": [review]}),
        json.dumps({"slot": "1", "supported": "true", "distinct": "True"}),
    ])
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: next(answers))
    result = quiz_generator.generate_quiz("BFS", evidence(), num_questions=1)
    assert result.generated_count == 1 and result.generation_error is None
    assert result.questions[0].sources == [1] and result.citations[0].label == 1


def test_rejection_logs_hide_raw_response_and_preserve_strict_semantics(caplog):
    malformed = '{"secret": "PRIVATE-MARKER"'
    with pytest.raises(LLMError, match="JSON"):
        decode_json(malformed, "quiz.candidate")
    assert "reason=parse_error" in caplog.text and "PRIVATE-MARKER" not in caplog.text
    draft = {**question(1), "slot": "PRIVATE-MARKER"}
    assert quiz_generator._candidates([draft], [1], 1) == []
    assert "field=slot type=int_type" in caplog.text and "PRIVATE-MARKER" not in caplog.text


def test_quote_normalization_preserves_numbers_operators_and_word_order():
    assert supports_quote("Insert the vertex into the priority queue.",
                          "“Insert the vertex into the priori-\nty queue!”")
    assert not supports_quote("The loop runs while i < 10.", "The loop runs while i <= 10.")
    assert not supports_quote("The loop runs while i < 10.", "The loop runs while i < 100.")
    assert supports_quote("The loop runs while i <= 10.", "The loop runs while i ≤ 10.")
    assert not supports_quote("The runtime is proportional to n2.", "The runtime is proportional to n².")
    assert not supports_quote("The vertex is not inserted into the queue.",
                              "The vertex is inserted into the queue.")


def test_topic_gate_rejects_topics_absent_from_documents():
    from app.orchestration.study_evidence import mentions_topic, topic_terms
    bfs = topic_terms("BFS")
    assert mentions_topic({"text": "Breadth-first search uses a queue."}, bfs)
    assert not mentions_topic({"text": "Merge sort splits the array in halves."}, bfs)
    assert mentions_topic({"text": "Thuật toán sắp xếp nổi bọt"}, topic_terms("sort"))
    assert not mentions_topic({"text": "Hash tables map keys"}, topic_terms("trie"))
    assert mentions_topic({"text": "Recursion is when a function calls itself."}, topic_terms("đệ quy"))


def test_vietnamese_topic_query_is_expanded_with_english_name():
    from app.orchestration.study_evidence import expanded_query
    assert expanded_query("đệ quy") == "đệ quy recursion"
    assert expanded_query("recursion") == "recursion"
    assert expanded_query("nấu phở") == "nấu phở"


@pytest.mark.parametrize(("aliases", "matching", "unrelated"), [
    (("sort", "sắp xếp", "sap xep"), "Mergesort splits an array.", "Assorted binary search exercises."),
    (("graph", "đồ thị", "do thi"), "Đồ thị có các đỉnh và cạnh.", "This paragraph covers arrays."),
    (("binary search", "tìm kiếm nhị phân"), "Binary search halves the search interval.",
     "A binary search tree stores keys at nodes."),
    (("AVL", "cây AVL"), "AVL rebalances by performing rotations.", "A heap maintains a partial order."),
])
def test_quiz_short_topic_aliases_filter_passage_body(aliases, matching, unrelated):
    from app.orchestration.study_evidence import quiz_topic_filter

    expected_search, _ = quiz_topic_filter(aliases[0])
    for alias in aliases:
        search, matches = quiz_topic_filter(alias)
        assert search == expected_search
        assert matches({"text": matching})
        assert not matches({"section": alias, "text": unrelated})


@pytest.mark.parametrize("alias", ["search", "searching", "tìm kiếm", "tim kiem"])
def test_general_search_aliases_require_general_or_comparative_evidence(alias):
    from app.orchestration.study_evidence import quiz_topic_filter

    search, matches = quiz_topic_filter(alias)
    assert search == quiz_topic_filter("search")[0]
    assert matches({"text": "Searching is the process of finding a key in a collection."})
    assert matches({"text": "Tìm kiếm là quá trình tìm một khóa trong tập dữ liệu."})
    assert matches({"text": "Linear search scans every item. Binary search splits the range."})
    assert matches({"text": "Các thuật toán tìm kiếm gồm tìm kiếm tuần tự và tìm kiếm nhị phân."})
    assert not matches({"section": "Searching", "text": "Binary search is an algorithm for searching sorted arrays."})
    assert not matches({"text": "Searching is efficient when a binary search tree is balanced."})
    assert not matches({"text": "Cây tìm kiếm nhị phân hỗ trợ tìm kiếm theo khóa."})
    assert not matches({"text": "Breadth-first search uses a queue."})
    assert not matches({"section": "Tìm kiếm nhị phân", "text": "Tìm kiếm tiếp ở nửa còn lại."})
    assert not matches({"text": "Research about stable sorting."})


@pytest.mark.parametrize("alias", ["binary search tree", "BST", "cây nhị phân tìm kiếm", "cây tìm kiếm nhị phân"])
def test_binary_search_tree_aliases_are_distinct_from_binary_search_algorithm(alias):
    from app.orchestration.study_evidence import quiz_topic_filter

    _, tree = quiz_topic_filter(alias)
    _, binary = quiz_topic_filter("binary search")
    tree_passage = {"text": "Binary search trees store ordered keys at nodes. Cây tìm kiếm nhị phân có hai cây con."}
    binary_passage = {"text": "Binary search halves the range. Tìm kiếm nhị phân yêu cầu mảng đã sắp xếp."}
    assert tree(tree_passage) and not tree(binary_passage)
    assert binary(binary_passage) and not binary(tree_passage)


@pytest.mark.parametrize("topic", ["sort, graph", "sort; graph", "sort；graph", "  "])
def test_quiz_request_requires_one_topic(topic):
    with pytest.raises(ValidationError):
        QuizRequest(topic=topic)
    assert QuizRequest.model_validate({"query": "sort", "difficulty": "old value"}).topic == "sort"


def test_rows_accepts_json_wrapper_variants():
    from app.generation.study_output import rows
    assert rows({"answers": [{"slot": 1}, {"slot": 2}]}, "t") == [{"slot": 1}, {"slot": 2}]
    assert rows({"1": {"correct_index": 0}, "2": {"slot": 2}}, "t") == [
        {"slot": "1", "correct_index": 0}, {"slot": 2}]
    assert rows({"checks": [{"slot": 1}]}, "t", "checks") == [{"slot": 1}]


def test_decode_json_recovers_embedded_value():
    assert decode_json('Here is the quiz:\n[{"slot": 1}]\nHope it helps', "t") == [{"slot": 1}]
    with pytest.raises(LLMError):
        decode_json("{broken json", "t")
