"""Build and independently verify batches of grounded DSA questions."""
import json
import logging
from difflib import SequenceMatcher
from uuid import uuid4

from pydantic import BaseModel, Field

from app.orchestration.study_evidence import normalized_text
from app.schemas.contracts import QuizQuestion, QuizResponse

from .abstention import EMPTY_WORKSPACE_RESPONSE, topic_missing_response
from .llm_client import ERROR_MESSAGES, LLMError, request_deadline
import time
from .rag_engine import generate_study_json, grounded_fields
from .study_output import normalize_row, reject, rows, supports_quote, validate_row

logger = logging.getLogger(__name__)

BATCH_SIZE = 5
MAX_QUESTIONS = 5
# Generation passes per request; each pass only drafts the slots still missing.
MAX_PASSES = 10
# Stop starting new passes when less than this many seconds remain.
PASS_RESERVE_SECONDS = 60
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
    stage = "quiz.candidate"
    payload = rows(payload, stage, "questions")
    accepted = {}
    kinds = {normalized_text(kind): kind for kind in blueprint.values()}
    free = [slot for slot in blueprint]
    for item in payload[:BATCH_SIZE]:
        item = normalize_row(item, integers=("slot", "correct_index"))
        if isinstance(item, dict):
            # Refill prompts ask for e.g. slot 5 only, but the model often
            # renumbers from 1. Map such rows onto the still-free slots in order.
            if item.get("slot") in free:
                free.remove(item["slot"])
            elif free and type(item.get("slot")) is int and item["slot"] not in accepted:
                item["slot"] = free.pop(0)
            # The scheduler owns slot type and difficulty; a label typo from the
            # model should not discard an otherwise valid, verified question.
            slot = item.get("slot")
            if type(slot) is int and slot in blueprint:
                item["question_type"] = blueprint[slot]
            elif isinstance(kind := item.get("question_type"), str):
                item["question_type"] = kinds.get(normalized_text(kind.replace("_", " ")), kind)
            item["difficulty"] = difficulty
        question = validate_row(Candidate, item, stage)
        if question is None:
            continue
        options = {normalized_text(option) for option in question.options}
        if question.slot not in blueprint:
            reason = "unassigned_slot"
        elif blueprint[question.slot] != question.question_type or question.difficulty != difficulty:
            reason = "blueprint_mismatch"
        elif len(options) != 4 or "" in options:
            reason = "duplicate_or_empty_options"
        elif len(set(question.sources)) != len(question.sources) or any(label > source_count for label in question.sources):
            reason = "invalid_citations"
        elif question.slot in accepted:
            reason = "duplicate_slot"
        else:
            accepted[question.slot] = question
            continue
        reject(stage, reason, slot=question.slot)
    logger.info("study_output stage=%s accepted=%d", stage, len(accepted))
    return list(accepted.values())


def _solved(payload, candidates, evidence, backup=None, missing=None):
    stage = "quiz.checker"
    payload = rows(payload, stage, "checks", "results")
    reports = {}
    conflicting = set()
    expected = {candidate.slot for candidate in candidates}
    for item in payload[:BATCH_SIZE]:
        item = normalize_row(item, integers=("slot", "correct_index", "source"),
                             booleans=("unique_answer", "on_topic"))
        check = validate_row(Verification, item, stage)
        if check is None:
            continue
        if check.slot not in expected:
            reject(stage, "unassigned_slot", slot=check.slot)
            continue
        if check.slot in reports:
            conflicting.add(check.slot)
        reports[check.slot] = check
    verified = []
    for question in candidates:
        report = reports.get(question.slot)
        if report is None:
            reason = "missing_report"
            if missing is not None:
                missing.append(question)
        elif question.slot in conflicting:
            reason = "duplicate_report"
        elif not report.unique_answer:
            reason = "answer_not_unique"
        elif not report.on_topic:
            reason = "off_topic"
        elif report.correct_index != question.correct_index:
            reason = "answer_mismatch"
        elif report.source not in question.sources:
            reason = "uncited_source"
        elif not supports_quote(report.quote, evidence[report.source - 1]["text"]):
            reject("quiz.quote", "unsupported_excerpt", slot=question.slot, source=report.source)
            if backup is not None:
                # The answer was independently confirmed; only the verbatim excerpt
                # failed. Keep it as a fallback to complete the requested count.
                backup.append(question)
            continue
        else:
            verified.append((question, report))
            continue
        reject(stage, reason, slot=question.slot)
    logger.info("study_output stage=quiz.quote accepted=%d", len(verified))
    return verified


def _approved_explanations(payload):
    stage = "quiz.explanation"
    payload = rows(payload, stage, "checks", "results")
    seen, approved = set(), set()
    for item in payload[:BATCH_SIZE]:
        item = normalize_row(item, integers=("slot",), booleans=("supported",))
        check = validate_row(ExplanationCheck, item, stage)
        if check is None:
            continue
        if check.slot in seen or not check.supported:
            reject(stage, "duplicate_report" if check.slot in seen else "unsupported_explanation", slot=check.slot)
            approved.discard(check.slot)
        else:
            approved.add(check.slot)
        seen.add(check.slot)
    return approved


def _verify_batch(query, candidates, evidence, backup=None):
    solved, missing = _check_answers(query, candidates, evidence, backup)
    # FIT-AI often answers only the first question of a batch; re-check the
    # unanswered ones one by one instead of discarding them.
    for candidate in missing:
        try:
            more, _ = _check_answers(query, [candidate], evidence, backup)
        except LLMError as error:
            if error.code not in {"invalid_json", "truncated_json"}:
                raise
            continue
        solved.extend(more)
    return _review_explanations(solved, evidence, backup)


def _check_answers(query, candidates, evidence, backup):
    # A whitelist keeps both the generator's key and explanation out of the solve prompt.
    input_fields = {"slot", "question", "options", "sources"}
    inputs = [candidate.model_dump(include=input_fields) for candidate in candidates]
    review = generate_study_json(
        json.dumps(inputs, ensure_ascii=False), evidence,
        f"Independently solve each question for topic {json.dumps(query, ensure_ascii=False)}. "
        "Use only its cited evidence. Reject unstated assumptions and ambiguity. Return a JSON array "
        "with slot and zero-based correct_index (0 is the first option) as numbers, "
        "unique_answer and on_topic as JSON booleans, "
        "source as a cited number, and quote copied in the original source language. Example shape only: "
        '[{"slot":1,"correct_index":0,"unique_answer":true,"on_topic":true,"source":1,"quote":"source excerpt"}].',
        task="quiz_verification", max_tokens=1600, stage="quiz.checker",
    )
    missing = []
    solved = _solved(review, candidates, evidence, backup, missing)
    return solved, missing


def _review_explanations(solved, evidence, backup):
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
        'Return only a JSON array shaped like [{"slot":1,"supported":true}], with a JSON number and boolean.',
        task="quiz_verification", max_tokens=600, stage="quiz.explanation",
    )
    approved = _approved_explanations(explanation_review)
    kept = [question for question, _ in solved if question.slot in approved]
    for question, _ in solved:
        if question.slot not in approved:
            reject("quiz.explanation", "not_approved", slot=question.slot)
            if backup is not None:
                backup.append(question)
    logger.info("study_output stage=quiz.explanation accepted=%d", len(kept))
    return kept


def _draft_batch(query, evidence, difficulty, blueprint, seen, *, attempt=0):
    proposal = generate_study_json(
        query, evidence,
        f"Draft a DSA quiz at {difficulty} difficulty. Slots and exact types: {json.dumps(blueprint)}. "
        "Easy tests recognition; Medium applies rules or traces short input; Hard tests multi-step reasoning. "
        "Cover distinct subtopics. Return a JSON array with slot, question, options (four plausible strings), "
        "correct_index (0-3, 0 is the first option), explanation, sources (one to three numeric labels), "
        "question_type, difficulty and topic. "
        'Encode slot and labels as JSON numbers, even though the assignment object has string keys. '
        'Example shape only: [{"slot":1,"question_type":"Concept","difficulty":"Easy","sources":[1],'
        '"correct_index":0,"question":"Question","options":["A","B","C","D"],"explanation":"Reason","topic":"DSA"}]. '
        "Use one correct answer. State assumptions for traces and complexity. Skip unsupported slots. "
        f"Avoid fill-in-the-blank questions and these previous questions: {json.dumps(seen, ensure_ascii=False)}."
        # FIT-AI caches identical prompts, so a refill would replay the same
        # rejected (or malformed) answer. A per-attempt marker forces a new draft.
        + (f" Attempt {attempt + 1} ({uuid4().hex[:8]}): write new questions on different facts." if attempt else ""),
        max_tokens=2400, task="quiz_generation", stage="quiz.candidate",
        # Deterministic first draft; refills need variety, otherwise an identical
        # prompt at temperature 0 returns the same rejected questions again.
        temperature=0.0 if attempt == 0 else 0.7,
    )
    return _candidates(proposal, blueprint, difficulty, len(evidence))


def generate_quiz(query, evidences, *, num_questions=5, difficulty="Medium", workspace_empty=False):
    if not evidences:
        return QuizResponse(
            query=query, answer=EMPTY_WORKSPACE_RESPONSE if workspace_empty else topic_missing_response(query),
            confidence_score=0.0, generation_mode="abstained", requested_count=num_questions,
        )
    num_questions = min(num_questions, MAX_QUESTIONS)
    kinds = BLUEPRINTS[difficulty]
    approved, seen, backup = {}, [], []
    failure = None
    try:
        # Refill passes, in bounded batches, only for slots still missing.
        for attempt in range(MAX_PASSES):
            pending = [slot for slot in range(1, num_questions + 1) if slot not in approved]
            if not pending:
                break
            deadline = request_deadline.get()
            if attempt and deadline is not None and deadline - time.monotonic() < PASS_RESERVE_SECONDS:
                break
            for start in range(0, len(pending), BATCH_SIZE):
                blueprint = {slot: kinds[(slot - 1) % len(kinds)]
                             for slot in pending[start:start + BATCH_SIZE]}
                try:
                    candidates = _draft_batch(query, evidences, difficulty, blueprint, seen, attempt=attempt)
                    checked = _verify_batch(query, candidates, evidences, backup) if candidates else []
                except LLMError as error:
                    if error.code not in {"invalid_json", "truncated_json"}:
                        raise
                    failure = error.code
                    continue
                for question in checked:
                    fingerprint = normalized_text(question.question)
                    if not fingerprint or any(SequenceMatcher(None, fingerprint, prior).ratio() >= .88 for prior in seen):
                        reject("quiz.final", "duplicate_or_empty_question", slot=question.slot)
                        continue
                    seen.append(fingerprint)
                    approved[question.slot] = QuizQuestion.model_validate(question.model_dump(exclude={"slot"}))
    except LLMError as error:
        failure = error.code

    # Complete the requested count with answer-verified fallbacks when the
    # strict excerpt/explanation checks left slots empty.
    for question in backup:
        missing = [slot for slot in range(1, num_questions + 1) if slot not in approved]
        if not missing:
            break
        fingerprint = normalized_text(question.question)
        if not fingerprint or any(SequenceMatcher(None, fingerprint, prior).ratio() >= .88 for prior in seen):
            continue
        seen.append(fingerprint)
        approved[missing[0]] = QuizQuestion.model_validate(question.model_dump(exclude={"slot"}))
        logger.info("study_output stage=quiz.fallback slot=%s", missing[0])

    questions = [approved[slot] for slot in sorted(approved)]

    generated = len(questions)
    logger.info("study_output stage=quiz.final accepted=%d requested=%d", generated, num_questions)
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
