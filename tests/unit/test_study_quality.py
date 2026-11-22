from copy import deepcopy
from datetime import date, timedelta
import json

import pytest
from pydantic import ValidationError

from app.generation import quiz_generator, rag_engine, revision_generator
from app.generation.study_planner import build_curriculum, schedule_topics, validate_plan
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
        "slot": slot, "correct_index": 0, "unique_answer": True, "explanation_supported": True,
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
        second, [verification(6, quote="The model supplied a sentence absent from the document.")],
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
    assert len(prompts) == 4 and "correct_index" not in json.loads(prompts[1][0])[0]
    assert "Algorithm Tracing" in prompts[2][1]


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


def provider_plan(catalog, slots, passages, *, exam_date, budget, level):
    days = {}
    labels = {topic.id: topic.sources[0] for topic in catalog}
    completed = []
    for i, (day, kind, topic, minutes) in enumerate(slots, 1):
        label = labels[topic]
        prior = [item for item in completed if item != topic]
        days.setdefault(day, []).append({
            "kind": kind, "topic_id": topic, "objective": f"Objective {i}",
            "knowledge": ["RAG-grounded study topic"], "activity": f"Activity {i}",
            "practice": f"Exercise {i}", "minutes": minutes, "prerequisites": prior[-1:],
            "sources": [label], "support_quote": passages[label - 1]["text"],
        })
        if kind == "learn":
            completed.append(topic)
    return {
        "exam_date": exam_date, "minutes_per_day": budget, "level": level,
        "topics": [topic.model_dump() for topic in catalog],
        "days": [{"date": day, "tasks": tasks} for day, tasks in days.items()],
    }


def test_study_date_inputs():
    body = RevisionRequest(all_topics=True, days_remaining=4)
    assert revision_generator.resolve_exam_date(body, today=TODAY) == (TODAY, TODAY + timedelta(days=4))
    with pytest.raises(ValueError):
        revision_generator.resolve_exam_date(RevisionRequest(all_topics=True, exam_date=TODAY), today=TODAY)
    with pytest.raises(ValidationError):
        RevisionRequest(query="Graph", days_remaining=4, exam_date=TODAY)
    with pytest.raises(ValidationError):
        RevisionRequest(days_remaining=4, minutes_per_day=0)


@pytest.mark.parametrize("problem", ["over_budget", "forward_prerequisite", "wrong_source",
                                     "made_up_quote", "duplicate_activity", "exam_day"])
def test_study_plan_validates_schedule_and_source_constraints(problem):
    curriculum, passages = build_curriculum(evidence())
    settings = dict(exam_date=TODAY + timedelta(days=5), budget=30, level="Beginner")
    chosen, slots = schedule_topics(curriculum, today=TODAY, **settings)
    valid = provider_plan(chosen, slots, passages, **settings)
    plan = validate_plan(valid, chosen, slots, passages, today=TODAY, **settings)
    assert plan and len(plan.topics) == 3
    assert {task.kind for day in plan.days for task in day.tasks} == {"learn", "review", "mock_test"}
    assert plan.days[1].tasks[0].prerequisites == ["T1"]
    broken = deepcopy(valid)
    first = broken["days"][0]["tasks"][0]
    if problem == "over_budget":
        first["minutes"] = 31
    elif problem == "forward_prerequisite":
        first["prerequisites"] = ["T2"]
    elif problem == "wrong_source":
        first["sources"] = [2]
    elif problem == "made_up_quote":
        first["support_quote"] = "This text does not exist in the document."
    elif problem == "duplicate_activity":
        broken["days"][1]["tasks"][0]["activity"] = first["activity"]
    else:
        broken["days"][-1]["date"] = settings["exam_date"]
    assert validate_plan(broken, chosen, slots, passages, today=TODAY, **settings) is None


def test_revision_reports_missing_coverage_and_does_not_fake_plan(monkeypatch):
    documents = evidence()
    settings = dict(exam_date=TODAY + timedelta(days=1), budget=15, level="Beginner")
    catalog, passages = build_curriculum(documents)
    chosen, slots = schedule_topics(catalog, today=TODAY, **settings)
    mock_plan = provider_plan(chosen, slots, passages, **settings)
    monkeypatch.setattr(revision_generator, "generate_study_json", lambda *args, **kwargs: mock_plan)
    revision = revision_generator.generate_revision("DSA", documents, today=TODAY, **settings)
    assert revision.plan.topics == chosen
    assert len(revision.uncovered_topics) == 2 and len(revision.warnings) == 2
    monkeypatch.setattr(revision_generator, "generate_study_json", lambda *args, **kwargs: None)
    unavailable = revision_generator.generate_revision("DSA", documents, today=TODAY, **settings)
    assert unavailable.plan is None and unavailable.generation_mode == "unavailable"
