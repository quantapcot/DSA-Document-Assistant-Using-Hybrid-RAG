from datetime import date, timedelta
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.generation import quiz_generator, rag_engine, revision_generator
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


def question(slot, kind, *, text=None, difficulty="Hard"):
    return {
        "slot": slot, "question": text or f"Which invariant determines the BFS traversal for case {slot}?",
        "options": ["distance", "vertex label", "stack size", "edge weight"],
        "correct_index": 0, "explanation": "The order follows distance from the start.",
        "sources": [1], "question_type": kind, "difficulty": difficulty, "topic": "BFS",
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


def test_quiz_blueprint_batches_and_filters(monkeypatch):
    kinds = quiz_generator.BLUEPRINTS["Hard"]
    first = [
        question(1, kinds[0], text="What determines BFS visit order?"),
        question(2, kinds[1], text="What determines BFS visit order?"),
        question(3, kinds[2], text="Which rule places newly discovered vertices into a queue?"),
        {**question(4, kinds[3]), "options": ["one", "Two", "two!", "three"]},
        {**question(5, kinds[4]), "sources": [99]},
    ]
    second = [question(6, kinds[5], text="How do you check that a vertex is reachable?")]
    values = iter([
        first, [verification(1), verification(2), verification(3, correct_index=1)],
        [{"slot": 1, "supported": True}, {"slot": 2, "supported": True}],
        second, [verification(6, quote="The model supplied a sentence absent from the document.")], [],
    ])
    prompts = []

    def mock(query, passages, instructions, **kwargs):
        prompts.append((query, instructions))
        return next(values)

    monkeypatch.setattr(quiz_generator, "generate_study_json", mock)
    quiz = quiz_generator.generate_quiz("BFS", evidence(), num_questions=6, difficulty="Hard")
    assert (quiz.generated_count, quiz.requested_count) == (1, 6)
    assert quiz.questions[0].question_type == "Algorithm Tracing"
    assert quiz.questions[0].difficulty == "Hard"
    assert quiz.citations[0].filename == "Chapter 1.pdf"
    assert len(prompts) == 6
    assert {"correct_index", "explanation"}.isdisjoint(json.loads(prompts[1][0])[0])
    assert "explanation" in json.loads(prompts[2][0])[0]
    assert "Algorithm Tracing" in prompts[3][1]
    assert '"1":' not in prompts[5][1]


def test_invalid_json_and_missing_evidence_are_explicit(monkeypatch):
    monkeypatch.setattr(rag_engine, "settings", type("Settings", (), {
        "LLM_BASE_URL": "https://fit.invalid/v1", "LLM_MODEL": "test", "OPENAI_API_KEY": True,
    }))
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: "{broken json")
    assert quiz_generator.generate_quiz("BFS", evidence()).generation_mode == "unavailable"
    result = quiz_generator.generate_quiz("BFS", [], workspace_empty=True)
    assert result.generation_mode == "abstained" and not result.questions
    for invalid in (0, 21):
        with pytest.raises(ValidationError):
            QuizRequest(query="BFS", num_questions=invalid)


@pytest.mark.parametrize(("count", "difficulty"), [(5, "Easy"), (10, "Medium"), (20, "Hard")])
def test_quiz_exact_count_and_refills_only_rejected_slots(monkeypatch, count, difficulty):
    prompts = [
        "Where are duplicate keys stored?", "Describe the visitation order.",
        "Can the root be removed safely?", "Pick the matching recurrence.",
        "Compare space utilization.", "Explain the role of parity.",
        "Which vertex is reachable?", "What changes after deletion?",
        "Find an invalid operation.", "Identify the loop invariant.",
        "When does rehashing happen?", "Choose a recursive base case.",
        "How is a median selected?", "Recognize a stable ordering.",
        "Estimate the amortized cost.", "Which path minimizes distance?",
        "Evaluate the balance condition.", "Name the required stack action.",
        "Select a suitable partition.", "Why use contiguous storage?",
    ]
    drafts = []
    reject_slot = 2

    def llm(query, passages, instructions, **kwargs):
        if "Slots and exact types:" in instructions:
            raw = instructions.split("Slots and exact types: ", 1)[1]
            assignments, _ = json.JSONDecoder().raw_decode(raw)
            assert len(assignments) <= 5
            drafts.append(list(map(int, assignments)))
            return [
                question(int(slot), kind, text=prompts[int(slot) - 1], difficulty=difficulty)
                for slot, kind in assignments.items()
            ]
        supplied = json.loads(query)
        if "Independently solve" in instructions:
            assert all("explanation" not in item and "correct_index" not in item for item in supplied)
            return [verification(item["slot"]) for item in supplied]
        return [
            {"slot": item["slot"], "supported": item["slot"] != reject_slot or drafts.count([reject_slot]) == 1}
            for item in supplied
        ]

    monkeypatch.setattr(quiz_generator, "generate_study_json", llm)
    result = quiz_generator.generate_quiz("BFS", evidence(), num_questions=count, difficulty=difficulty)
    assert result.requested_count == result.generated_count == count
    assert drafts[-1] == [reject_slot]
    assert {question.question_type for question in result.questions} == {
        "Concept", "Complexity", "Algorithm Tracing", "Comparison", "Application",
    }


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


def test_quiz_one_easy_accepts_unambiguous_qwen_representations(configured_fit, monkeypatch):
    draft = {
        **question(1, "Concept", difficulty="Easy"), "slot": "1",
        "question_type": " concept ", "difficulty": "easy", "sources": ["[1]"],
    }
    review = verification(
        "1", correct_index="0", source="[1]", unique_answer="true", on_topic="True",
        quote="“BREADTH-FIRST SEARCH visits vertices in ascending order of distance from the start!”",
    )
    answers = iter([
        "<think>reasoning</think>\n```json\n" + json.dumps({"questions": [draft]}) + "\n```",
        json.dumps({"checks": [review]}),
        json.dumps({"slot": "1", "supported": "true"}),
    ])
    monkeypatch.setattr(rag_engine, "generate_answer", lambda *args, **kwargs: next(answers))
    result = quiz_generator.generate_quiz("BFS", evidence(), num_questions=1, difficulty="Easy")
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
    draft = {**question(1, "Concept", difficulty="Easy"), "slot": "PRIVATE-MARKER"}
    assert quiz_generator._candidates([draft], {1: "Concept"}, "Easy", 1) == []
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
