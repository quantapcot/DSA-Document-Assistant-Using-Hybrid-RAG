"""Build and independently verify batches of grounded DSA questions."""
import json
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from app.orchestration.study_evidence import normalized_text
from app.schemas.contracts import QuizQuestion, QuizResponse

from .abstention import ABSTENTION_RESPONSE, EMPTY_WORKSPACE_RESPONSE
from .llm_client import ERROR_MESSAGES, LLMError
from .rag_engine import generate_study_json, grounded_fields

BATCH_SIZE = 5
BLUEPRINTS = {
    "Easy": ("Concept", "Application", "Complexity", "Comparison", "Algorithm Tracing", "Concept"),
    "Medium": ("Concept", "Complexity", "Algorithm Tracing", "Comparison", "Application"),
    "Hard": ("Algorithm Tracing", "Complexity", "Application", "Comparison", "Concept", "Algorithm Tracing"),
}


class Candidate(QuizQuestion):
    slot: int = Field(strict=True, ge=1)


class Verification(BaseModel):
    slot: int = Field(strict=True, ge=1)
    correct_index: int = Field(strict=True, ge=0, le=3)
    unique_answer: bool = Field(strict=True)
    on_topic: bool = Field(strict=True)
    source: int = Field(strict=True, ge=1)
    quote: str = Field(min_length=20)


class ExplanationCheck(BaseModel):
    slot: int = Field(strict=True, ge=1)
    supported: bool = Field(strict=True)


def _candidates(payload, blueprint, difficulty, source_count):
    if not isinstance(payload, list):
        return []
    accepted = {}
    for item in payload[:BATCH_SIZE]:
        try:
            question = Candidate.model_validate(item)
        except ValueError:
            continue
        options = {normalized_text(option) for option in question.options}
        if (
            blueprint.get(question.slot) == question.question_type
            and question.difficulty == difficulty
            and len(options) == 4 and "" not in options
            and len(set(question.sources)) == len(question.sources)
            and all(1 <= label <= source_count for label in question.sources)
        ):
            accepted.setdefault(question.slot, question)
    return list(accepted.values())


def _solved(payload, candidates, evidence):
    if not isinstance(payload, list):
        return []
    reports = {}
    conflicting = set()
    for item in payload[:BATCH_SIZE]:
        try:
            check = Verification.model_validate(item)
        except ValueError:
            continue
        if check.slot in reports:
            conflicting.add(check.slot)
        reports[check.slot] = check
    verified = []
    for question in candidates:
        report = reports.get(question.slot)
        if (report is None or question.slot in conflicting or not report.unique_answer
                or not report.on_topic or report.correct_index != question.correct_index
                or report.source not in question.sources):
            continue
        quote = " ".join(report.quote.split())
        passage = " ".join(evidence[report.source - 1]["text"].split())
        if len(quote) >= 20 and quote in passage:
            verified.append((question, report))
    return verified


def _approved_explanations(payload):
    if not isinstance(payload, list):
        return set()
    seen, approved = set(), set()
    for item in payload[:BATCH_SIZE]:
        try:
            check = ExplanationCheck.model_validate(item)
        except ValueError:
            continue
        if check.slot in seen or not check.supported:
            approved.discard(check.slot)
        else:
            approved.add(check.slot)
        seen.add(check.slot)
    return approved


def _verify_batch(query, candidates, evidence):
    # A whitelist keeps both the generator's key and explanation out of the solve prompt.
    input_fields = {"slot", "question", "options", "sources"}
    inputs = [candidate.model_dump(include=input_fields) for candidate in candidates]
    review = generate_study_json(
        json.dumps(inputs, ensure_ascii=False), evidence,
        f"Independently solve each question for topic {json.dumps(query, ensure_ascii=False)}. "
        "Use only its cited evidence. Reject unstated assumptions and ambiguity. Return a JSON array "
        "with slot, correct_index, unique_answer, on_topic, source (a cited label), and a verbatim quote.",
        task="quiz_verification", max_tokens=1600,
    )
    solved = _solved(review, candidates, evidence)
    if not solved:
        return []
    checks = [{
        "slot": question.slot, "question": question.question,
        "verified_answer": question.options[report.correct_index],
        "explanation": question.explanation, "sources": question.sources,
        "verified_quote": report.quote,
    } for question, report in solved]
    explanation_review = generate_study_json(
        json.dumps(checks, ensure_ascii=False), evidence,
        "Audit each explanation against the verified answer and its cited evidence. "
        "Set supported to false for factual errors, contradictions or unsupported reasoning. "
        "Return only a JSON array with slot and supported (boolean).",
        task="quiz_verification", max_tokens=600,
    )
    approved = _approved_explanations(explanation_review)
    return [question for question, _ in solved if question.slot in approved]


def _draft_batch(query, evidence, difficulty, blueprint, seen):
    proposal = generate_study_json(
        query, evidence,
        f"Draft a DSA quiz at {difficulty} difficulty. Slots and exact types: {json.dumps(blueprint)}. "
        "Easy tests recognition; Medium applies rules or traces short input; Hard tests multi-step reasoning. "
        "Cover distinct subtopics. Return a JSON array with slot, question, options (four plausible strings), "
        "correct_index (0-3), explanation, sources (one to three labels), question_type, difficulty and topic. "
        "Use one correct answer. State assumptions for traces and complexity. Skip unsupported slots. "
        f"Avoid fill-in-the-blank questions and these previous questions: {json.dumps(seen, ensure_ascii=False)}.",
        max_tokens=2400, task="quiz_generation",
    )
    return _candidates(proposal, blueprint, difficulty, len(evidence))


def generate_quiz(query, evidences, *, num_questions=5, difficulty="Medium", workspace_empty=False):
    if not evidences:
        return QuizResponse(
            query=query, answer=EMPTY_WORKSPACE_RESPONSE if workspace_empty else ABSTENTION_RESPONSE,
            confidence_score=0.0, generation_mode="abstained", requested_count=num_questions,
        )
    kinds = BLUEPRINTS[difficulty]
    approved, seen = {}, []
    failure = None
    try:
        # One refill pass, in bounded batches, only for slots still missing.
        for _ in range(2):
            pending = [slot for slot in range(1, num_questions + 1) if slot not in approved]
            for start in range(0, len(pending), BATCH_SIZE):
                blueprint = {slot: kinds[(slot - 1) % len(kinds)]
                             for slot in pending[start:start + BATCH_SIZE]}
                try:
                    candidates = _draft_batch(query, evidences, difficulty, blueprint, seen)
                    checked = _verify_batch(query, candidates, evidences) if candidates else []
                except LLMError as error:
                    if error.code not in {"invalid_json", "truncated_json"}:
                        raise
                    failure = error.code
                    continue
                for question in checked:
                    fingerprint = normalized_text(question.question)
                    if not fingerprint or any(SequenceMatcher(None, fingerprint, prior).ratio() >= .88 for prior in seen):
                        continue
                    seen.append(fingerprint)
                    approved[question.slot] = QuizQuestion.model_validate(question.model_dump(exclude={"slot"}))
    except LLMError as error:
        failure = error.code

    questions = [approved[slot] for slot in sorted(approved)]

    generated = len(questions)
    if not generated:
        return QuizResponse(
            query=query, answer=ERROR_MESSAGES[failure or "invalid_output"],
            confidence_score=0.0, generation_mode="unavailable", requested_count=num_questions,
            generation_error=failure or "invalid_output",
        )
    answer = f"Đã tạo {generated}/{num_questions} câu hỏi đạt kiểm tra."
    fields = grounded_fields(query, answer, evidences, "hybrid_rerank", "llm",
                             [label for question in questions for label in question.sources])
    if generated < num_questions and failure:
        fields["answer"] += " " + ERROR_MESSAGES[failure]
    return QuizResponse(**fields, questions=questions, requested_count=num_questions, generated_count=generated,
                        generation_error=failure if generated < num_questions else None)
