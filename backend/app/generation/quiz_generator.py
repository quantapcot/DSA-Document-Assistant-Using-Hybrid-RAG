"""Draft and verify one grounded quiz for a single topic."""
import json
import logging
from collections import Counter
from difflib import SequenceMatcher
from time import monotonic
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.config import settings
from app.orchestration.study_evidence import normalized_text
from app.schemas.contracts import QuizQuestion, QuizResponse

from .abstention import EMPTY_WORKSPACE_RESPONSE, topic_missing_response
from .llm_client import ERROR_MESSAGES, LLMError, quiz_provider_calls, request_deadline
from .rag_engine import generate_study_json, grounded_fields
from .study_output import normalize_row, reject, rows, supports_quote, validate_row

logger = logging.getLogger(__name__)
MAX_PASSES = 3
MIN_REFILL_SECONDS = 10
RETRYABLE_OUTPUT = {"invalid_json", "truncated_json"}


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
    distinct: bool = Field(strict=True)


def _ask(calls, stage, query, evidence, instructions, **options):
    if monotonic() >= request_deadline.get():
        raise LLMError("timeout")
    calls[stage] += 1
    started = monotonic()
    try:
        return generate_study_json(query, evidence, instructions, stage=stage, **options)
    finally:
        logger.info("quiz stage=%s call=%d elapsed_ms=%.1f", stage, calls[stage],
                    (monotonic() - started) * 1000)


def _candidates(payload, pending, topic, source_count):
    stage = "quiz.candidate"
    started = monotonic()
    normalized = [normalize_row(item, integers=("slot", "correct_index"))
                  for item in rows(payload, stage, "questions")[:len(pending)]]
    slots = [item.get("slot") if isinstance(item, dict) else None for item in normalized]
    # Some refill responses number the entire batch from 1. Only remap a
    # complete sequential numbering, never an ambiguous or repeated slot.
    if (all(type(slot) is int for slot in slots) and slots == list(range(1, len(slots) + 1))
            and not set(slots) <= set(pending)):
        for item, slot in zip(normalized, pending):
            item["slot"] = slot
    accepted = {}
    for item in normalized:
        if isinstance(item, dict):
            item["topic"] = topic
        question = validate_row(Candidate, item, stage)
        if question is None:
            continue
        options = {normalized_text(option) for option in question.options}
        if question.slot not in pending:
            reason = "unassigned_slot"
        elif question.slot in accepted:
            reason = "duplicate_slot"
        elif not normalized_text(question.question) or len(options) != 4 or "" in options:
            reason = "empty_question_or_non_distinct_options"
        elif len(set(question.sources)) != len(question.sources) or max(question.sources) > source_count:
            reason = "invalid_citations"
        else:
            accepted[question.slot] = question
            continue
        reject(stage, reason, slot=question.slot)
    for slot in pending:
        if slot not in accepted:
            reject(stage, "no_valid_candidate", slot=slot)
    logger.info("quiz stage=candidate_validation accepted=%d elapsed_ms=%.1f",
                len(accepted), (monotonic() - started) * 1000)
    return list(accepted.values())


def _reports(payload, slots, model, stage, *, integers, booleans):
    reports, seen = {}, set()
    for item in rows(payload, stage, "checks", "results"):
        item = normalize_row(item, integers=integers, booleans=booleans)
        check = validate_row(model, item, stage)
        if check is None:
            continue
        if check.slot not in slots:
            reject(stage, "unassigned_slot", slot=check.slot)
        elif check.slot in seen:
            reject(stage, "duplicate_report", slot=check.slot)
            reports.pop(check.slot, None)
        else:
            seen.add(check.slot)
            reports[check.slot] = check
    return reports


def _cited_passages(evidence, candidates):
    labels = sorted({label for question in candidates for label in question.sources})
    return {label: evidence[label - 1] for label in labels}


def _check_answers(topic, candidates, evidence, calls):
    stage = "quiz.checker"
    # Only the stem, choices and source labels go to the independent solver.
    fields = {"slot", "question", "options", "sources"}
    inputs = [candidate.model_dump(include=fields) for candidate in candidates]
    review = _ask(
        calls, stage, json.dumps(inputs, ensure_ascii=False), _cited_passages(evidence, candidates),
        f"Independently solve every question for topic {json.dumps(topic, ensure_ascii=False)}. "
        "Use only each question's cited sources. Check every option, including ambiguity or unstated assumptions. "
        "For every slot return integers slot, correct_index (zero-based) and source (a cited label), "
        "JSON booleans unique_answer and on_topic, and a short contiguous quote from that source in its "
        'original language. Shape: [{"slot":1,"correct_index":0,"unique_answer":true,"on_topic":true,'
        '"source":1,"quote":"source excerpt"}]. Echo the input slot numbers.',
        task="quiz_verification", max_tokens=1600,
    )
    checks = _reports(review, {candidate.slot for candidate in candidates}, Verification, stage,
                      integers=("slot", "correct_index", "source"), booleans=("unique_answer", "on_topic"))
    started = monotonic()
    solved = []
    for question in candidates:
        report = checks.get(question.slot)
        if report is None:
            reason = "missing_report"
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
            continue
        else:
            solved.append((question, report))
            continue
        reject(stage, reason, slot=question.slot)
    logger.info("quiz stage=quote accepted=%d elapsed_ms=%.1f", len(solved), (monotonic() - started) * 1000)
    return solved


def _review_explanations(solved, evidence, approved, calls):
    if not solved:
        return []
    stage = "quiz.explanation"
    candidates = [question for question, _ in solved]
    inputs = [{
        "slot": question.slot, "question": question.question, "options": question.options,
        "verified_answer": question.options[report.correct_index],
        "explanation": question.explanation, "sources": question.sources,
        "verified_quote": report.quote,
    } for question, report in solved]
    previous = [{"question": question.question, "answer": question.options[question.correct_index]}
                for question in approved.values()]
    review = _ask(
        calls, stage, json.dumps({"questions": inputs, "previous": previous}, ensure_ascii=False),
        _cited_passages(evidence, candidates),
        "Audit every question. Set supported to true only if its explanation justifies the verified answer "
        "without unsupported claims and each cited source supports the question or explanation. "
        "Set distinct to false if it tests the same fact or reasoning as a previous question, even if "
        "paraphrased, translated, reversed or reordering the options. For redundant questions in this "
        "batch, mark at most one distinct. Different facts about the same topic are allowed. "
        'Return only a JSON array such as [{"slot":1,"supported":true,"distinct":true}]. Echo all input slots.',
        task="quiz_verification", max_tokens=600,
    )
    reports = _reports(review, {question.slot for question in candidates}, ExplanationCheck, stage,
                       integers=("slot",), booleans=("supported", "distinct"))
    kept = []
    for question in candidates:
        report = reports.get(question.slot)
        if report is None:
            reason = "missing_report"
        elif not report.supported:
            reason = "unsupported_explanation"
        elif not report.distinct:
            reason = "repeated_question"
        else:
            kept.append(question)
            continue
        reject(stage, reason, slot=question.slot)
    return kept


def _draft_batch(topic, evidence, pending, approved, calls, attempt):
    instructions = (
        f"Make a quiz about the student's single topic. Slots: {json.dumps(pending)}. "
        "Draft one distinct question per slot when the sources support it. Skip unsupported slots. "
        "Use four distinct, plausible options with exactly one correct answer. Cite one to three source labels "
        "and explain why the answer is correct based on those sources. Use concise wording and avoid fill-in-the-blank questions. "
        "Return one JSON array. Copy slot numbers exactly. Shape only: "
        '[{"slot":1,"question":"Question","options":["A","B","C","D"],"correct_index":0,'
        '"explanation":"Reason","sources":[1]}]. correct_index is zero-based. '
        f"Ask about different facts than these accepted questions: "
        f"{json.dumps([question.question for question in approved.values()], ensure_ascii=False)}."
    )
    if attempt:
        # New draft prompts avoid replaying a cached answer on an identical refill.
        instructions += f" Refill {attempt} ({uuid4().hex[:8]}). Use other facts and questions for these slots."
    payload = _ask(calls, "quiz.candidate", topic, evidence, instructions, task="quiz_generation",
                   max_tokens=2400, temperature=0.7 if attempt else 0.0)
    return _candidates(payload, pending, topic, len(evidence))


def _verified_questions(topic, evidence, count, calls):
    approved, fingerprints = {}, []
    failure = None
    for attempt in range(MAX_PASSES):
        pending = [slot for slot in range(1, count + 1) if slot not in approved]
        if not pending:
            break
        if attempt and request_deadline.get() - monotonic() < MIN_REFILL_SECONDS:
            failure = "timeout"
            break
        logger.info("quiz pass=%d pending=%d", attempt + 1, len(pending))
        try:
            candidates = _draft_batch(topic, evidence, pending, approved, calls, attempt)
            solved = _check_answers(topic, candidates, evidence, calls) if candidates else []
            checked = _review_explanations(solved, evidence, approved, calls)
        except LLMError as error:
            failure = error.code
            reject("quiz.pass", error.code)
            if error.code in RETRYABLE_OUTPUT:
                continue
            break
        failure = None
        for question in checked:
            fingerprint = normalized_text(question.question)
            if any(SequenceMatcher(None, fingerprint, prior).ratio() >= .90 for prior in fingerprints):
                reject("quiz.final", "duplicate_question", slot=question.slot)
                continue
            fingerprints.append(fingerprint)
            approved[question.slot] = QuizQuestion.model_validate(question.model_dump(exclude={"slot"}))
    return [approved[slot] for slot in sorted(approved)], failure


def generate_quiz(topic, evidences, *, num_questions=5, workspace_empty=False):
    started = monotonic()
    if not evidences:
        reason = EMPTY_WORKSPACE_RESPONSE if workspace_empty else topic_missing_response(topic)
        logger.info("quiz stage=final accepted=0 requested=%d fit_ai_calls=0 reason=no_evidence", num_questions)
        return QuizResponse(
            query=topic, answer=f"Chỉ tạo được 0/{num_questions} câu đạt kiểm tra. {reason}",
            confidence_score=0.0, generation_mode="abstained", requested_count=num_questions,
        )

    deadline = request_deadline.get()
    if deadline is None:
        deadline = started + settings.QUIZ_REQUEST_TIMEOUT
    # Leave time to report already verified questions before the HTTP deadline.
    margin = min(2.0, max(0.0, deadline - started) * .1)
    deadline_token = request_deadline.set(deadline - margin)
    provider_token = quiz_provider_calls.set(0)
    calls = Counter()
    try:
        questions, failure = _verified_questions(topic, evidences, num_questions, calls)
    finally:
        provider_calls = quiz_provider_calls.get()
        quiz_provider_calls.reset(provider_token)
        request_deadline.reset(deadline_token)

    generated = len(questions)
    if generated < num_questions:
        reason = ERROR_MESSAGES[failure] if failure else (
            "Tài liệu hoặc đầu ra FIT-AI chưa đủ để xác minh thêm câu hỏi."
        )
        answer = f"Chỉ tạo được {generated}/{num_questions} câu đạt kiểm tra. {reason}"
        failure = failure or "invalid_output"
    else:
        answer = f"Đã tạo đủ {generated}/{num_questions} câu đạt kiểm tra."
    logger.info("quiz stage=final accepted=%d requested=%d llm_requests=%d fit_ai_calls=%d "
                "elapsed_ms=%.1f reason=%s", generated, num_questions, sum(calls.values()), provider_calls,
                (monotonic() - started) * 1000, failure)
    if not questions:
        return QuizResponse(query=topic, answer=answer, confidence_score=0.0, generation_mode="unavailable",
                            requested_count=num_questions, generation_error=failure)
    labels = [label for question in questions for label in question.sources]
    fields = grounded_fields(topic, answer, evidences, "hybrid_rerank", "llm", labels)
    return QuizResponse(**fields, questions=questions, requested_count=num_questions, generated_count=generated,
                        generation_error=failure)
