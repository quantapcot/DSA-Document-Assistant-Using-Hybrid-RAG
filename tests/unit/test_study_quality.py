from datetime import date, timedelta
import json
import logging
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.generation import llm_client, quiz_generator, rag_engine, revision_generator
from app.generation.llm_client import LLMError
from app.generation.study_output import decode_json, supports_quote
from app.generation.study_planner import (
    assemble_plan, build_curriculum, schedule_topics, validate_batch,
)
from app.orchestration.study_evidence import diverse_evidence
from app.schemas.contracts import QuizRequest, RevisionRequest

TODAY = date(2026, 11, 22)
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
    monkeypatch.setattr(rag_engine, "settings", type("Settings", (), {
        "LLM_BASE_URL": "https://fit.invalid/v1", "LLM_MODEL": "test", "OPENAI_API_KEY": True,
    }))
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
    config = Settings(_env_file=None, LLM_BASE_URL="https://fit.invalid/v1",
                      LLM_MODEL="qwen-fixture", OPENAI_API_KEY="PRIVATE-MARKER")
    monkeypatch.setattr(rag_engine, "settings", config)
    monkeypatch.setattr(llm_client, "settings", config)
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)
    attempts = []

    def call(*args, timeout, **kwargs):
        attempts.append(timeout)
        raise TimeoutError("PRIVATE-MARKER")

    monkeypatch.setattr(llm_client, "_call_openai", call)
    with caplog.at_level(logging.INFO):
        result = quiz_generator.generate_quiz("BFS", evidence(), num_questions=1)
    assert attempts == [config.LLM_QUIZ_GENERATION_TIMEOUT] * 3
    assert result.generation_error == "timeout" and not result.questions
    assert "0/1" in result.answer
    assert "llm_requests=1 fit_ai_calls=3" in caplog.text and "elapsed_ms=" in caplog.text
    assert "PRIVATE-MARKER" not in caplog.text


def task_contents(slots, catalog, passages):
    labels = {topic.id: topic.sources[0] for topic in catalog}
    return [{
        "slot": slot.id, "objective": f"Objective {slot.id}",
        "knowledge": ["A sourced study topic"], "activity": f"Activity {slot.id}",
        "practice": f"Exercise {slot.id}", "sources": [labels[topic] for topic in slot.topic_ids],
        "support_quote": passages[labels[slot.topic_ids[0]] - 1]["text"],
    } for slot in slots]


def test_study_date_inputs():
    body = RevisionRequest(all_topics=True, days_remaining=4)
    assert revision_generator.resolve_exam_date(body, today=TODAY) == (TODAY, TODAY + timedelta(days=4))
    with pytest.raises(ValueError):
        revision_generator.resolve_exam_date(RevisionRequest(all_topics=True, exam_date=TODAY), today=TODAY)
    with pytest.raises(ValidationError):
        RevisionRequest(query="Graph", days_remaining=4, exam_date=TODAY)
    with pytest.raises(ValidationError):
        RevisionRequest(days_remaining=4, minutes_per_day=0)


@pytest.mark.parametrize("problem", ["injected_minutes", "injected_prerequisite", "wrong_source", "made_up_quote"])
def test_study_batch_rejects_schedule_changes_and_bad_sources(problem):
    catalog, passages = build_curriculum(evidence())
    settings = dict(exam_date=TODAY + timedelta(days=5), budget=30, level="Beginner")
    chosen, slots = schedule_topics(catalog, passages, today=TODAY, **settings)
    batch = task_contents(slots, chosen, passages)
    assert validate_batch(batch, slots, chosen, passages)
    first = batch[0]
    if problem == "injected_minutes":
        first["minutes"] = 200
    elif problem == "injected_prerequisite":
        first["prerequisites"] = ["T2"]
    elif problem == "wrong_source":
        first["sources"] = [2]
    else:
        first["support_quote"] = "This text does not exist in the document."
    assert validate_batch(batch, slots, chosen, passages) is None


def test_revision_retries_truncated_batch_and_drops_failed_slots(monkeypatch):
    documents = evidence()
    settings = dict(exam_date=TODAY + timedelta(days=5), budget=30, level="Beginner")
    catalog, passages = build_curriculum(documents)
    chosen, slots = schedule_topics(catalog, passages, today=TODAY, **settings)
    payloads = {item["slot"]: item for item in task_contents(slots, chosen, passages)}
    calls = []

    def fake_model(query, sources, instructions, **kwargs):
        assignments, _ = json.JSONDecoder().raw_decode(instructions.split("Assignments: ", 1)[1])
        ids = [item["slot"] for item in assignments]
        calls.append(ids)
        assert kwargs["max_tokens"] <= 4000 and kwargs["task"] == "revision"
        if calls == [[1, 2, 3]]:
            raise LLMError("truncated_json")
        return [payloads[i] for i in ids]

    monkeypatch.setattr(revision_generator, "generate_study_json", fake_model)
    response = revision_generator.generate_revision("DSA", documents, today=TODAY, **settings)
    assert response.plan and calls[:4] == [[1, 2, 3], [1], [2], [3]]

    def bad_model(query, sources, instructions, **kwargs):
        assignments, _ = json.JSONDecoder().raw_decode(instructions.split("Assignments: ", 1)[1])
        ids = [item["slot"] for item in assignments]
        if ids == [1, 2, 3]:
            return [payloads[i] for i in ids]
        raise LLMError("invalid_json")

    monkeypatch.setattr(revision_generator, "generate_study_json", bad_model)
    response = revision_generator.generate_revision("DSA", documents, today=TODAY, **settings)
    # Failed later slots are dropped; the plan keeps the sessions that passed.
    assert response.plan is not None
    assert any("Bỏ" in warning for warning in response.warnings)

    monkeypatch.setattr(revision_generator, "generate_study_json",
                        lambda *args, **kwargs: (_ for _ in ()).throw(LLMError("invalid_json")))
    response = revision_generator.generate_revision("DSA", documents, today=TODAY, **settings)
    assert response.plan is None and response.generation_error == "invalid_output"


def test_revision_coverage_dependencies_and_varying_minutes(monkeypatch):
    chunks = [
        {**evidence()[0], "section": None, "text": "BFS requires a queue. Complexity and invariants matter."},
        {**evidence()[0], "section": None, "chunk_id": "queue", "page": 2, "text": "Queues keep inserted data in first-in first-out order."},
        *[{**item, "document_id": "extra-" + item["chunk_id"]}
          for item in evidence(15)],
    ]
    catalog, passages = build_curriculum(chunks)
    assert len(catalog) == 17
    monkeypatch.setattr("app.generation.study_planner.settings.REVISION_MAX_TOPICS", 14)
    settings = dict(exam_date=TODAY + timedelta(days=12), budget=50, level="Intermediate")
    chosen, slots = schedule_topics(catalog, passages, today=TODAY, **settings)
    learn = [slot for slot in slots if slot.kind == "learn"]
    assert len(chosen) == 14 and learn[0].topic_ids == ("T2",)
    assert learn[1].prerequisites == ("T2",)
    assert learn[0].minutes < learn[1].minutes
    assert len({t for s in slots if s.kind == "mock_test" for t in s.topic_ids}) > 4
    batch = validate_batch(task_contents(slots, chosen, passages), slots, chosen, passages)
    assert assemble_plan(batch, chosen, slots, passages, today=TODAY, **settings)
    batch[slots[1].id].activity = batch[slots[0].id].activity
    assert assemble_plan(batch, chosen, slots, passages, today=TODAY, **settings) is None

    monkeypatch.setattr("app.generation.study_planner.settings.REVISION_SOURCE_TOKEN_BUDGET", 300)
    short, short_slots = schedule_topics(catalog, passages, today=TODAY,
        exam_date=TODAY + timedelta(days=1), budget=15, level="Beginner")
    assert len(short) == 1 and short_slots[0].minutes == 15
    monkeypatch.setattr(revision_generator, "generate_study_json",
        lambda *args, **kwargs: task_contents(short_slots, short, passages))
    result = revision_generator.generate_revision("DSA", chunks, today=TODAY,
        exam_date=TODAY + timedelta(days=1), budget=15, level="Beginner")
    assert result.plan and len(result.uncovered_topics) == 16


@pytest.fixture
def configured_fit(monkeypatch):
    monkeypatch.setattr(rag_engine, "settings", SimpleNamespace(
        LLM_BASE_URL="https://fit.invalid/v1", LLM_MODEL="qwen-fixture", OPENAI_API_KEY=True,
    ))


def test_quiz_one_accepts_unambiguous_qwen_representations(configured_fit, monkeypatch):
    draft = {
        **question(1), "slot": "1", "sources": ["[1]"],
    }
    review = verification(
        "1", correct_index="0", source="[1]", unique_answer="true", on_topic="True",
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


def test_short_revision_accepts_source_strings_and_checks_quote(configured_fit, monkeypatch, caplog):
    documents = evidence(1)
    exam_date = TODAY + timedelta(days=1)
    kwargs = dict(today=TODAY, exam_date=exam_date, budget=15, level="Beginner")
    catalog, passages = build_curriculum(documents)
    chosen, slots = schedule_topics(catalog, passages, **kwargs)
    content = task_contents(slots, chosen, passages)[0]
    content.update(slot="1", sources=["[1]"], support_quote=QUOTE.upper().rstrip(".") + "!")
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **options: json.dumps({"tasks": [content]}))
    result = revision_generator.generate_revision("BFS", documents, **kwargs)
    assert result.plan is not None and len(result.plan.days) == 1
    assert result.plan.days[0].tasks[0].sources == [1]
    content["support_quote"] = "PRIVATE-MARKER: a stack is the only valid data structure"
    result = revision_generator.generate_revision("BFS", documents, **kwargs)
    assert result.plan is None and result.generation_error == "invalid_output"
    assert "stage=revision.quote reason=unsupported_excerpt slot=1" in caplog.text
    assert "PRIVATE-MARKER" not in caplog.text


def test_rejection_logs_hide_raw_response_and_preserve_strict_semantics(caplog):
    malformed = '{"secret": "PRIVATE-MARKER"'
    with pytest.raises(LLMError, match="JSON"):
        decode_json(malformed, "quiz.candidate")
    assert "reason=parse_error" in caplog.text and "PRIVATE-MARKER" not in caplog.text
    draft = {**question(1), "slot": "PRIVATE-MARKER"}
    assert quiz_generator._candidates([draft], [1], "BFS", 1) == []
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


def test_rows_accepts_fit_ai_wrapper_variants():
    from app.generation.study_output import rows
    assert rows({"answers": [{"slot": 1}, {"slot": 2}]}, "t") == [{"slot": 1}, {"slot": 2}]
    assert rows({"1": {"correct_index": 0}, "2": {"slot": 2}}, "t") == [
        {"slot": "1", "correct_index": 0}, {"slot": 2}]
    assert rows({"checks": [{"slot": 1}]}, "t", "checks") == [{"slot": 1}]


def test_decode_json_recovers_embedded_value():
    assert decode_json('Here is the quiz:\n[{"slot": 1}]\nHope it helps', "t") == [{"slot": 1}]
    with pytest.raises(LLMError):
        decode_json("{broken json", "t")
