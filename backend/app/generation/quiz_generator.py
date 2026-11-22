"""Build and independently verify batches of grounded DSA questions."""
import json
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from app.orchestration.study_evidence import normalized_text
from app.schemas.contracts import QuizQuestion, QuizResponse

from .abstention import ABSTENTION_RESPONSE, EMPTY_WORKSPACE_RESPONSE
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
    explanation_supported: bool = Field(strict=True)
    source: int = Field(strict=True, ge=1)
    quote: str = Field(min_length=20)


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


def _verified(payload, candidates, evidence):
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
                or not report.explanation_supported or report.correct_index != question.correct_index
                or report.source not in question.sources):
            continue
        quote = " ".join(report.quote.split())
        passage = " ".join(evidence[report.source - 1]["text"].split())
        if len(quote) >= 20 and quote in passage:
            verified.append(QuizQuestion.model_validate(question.model_dump(exclude={"slot"})))
    return verified


def generate_quiz(query, evidences, *, num_questions=5, difficulty="Medium", workspace_empty=False):
    if not evidences:
        return QuizResponse(
            query=query, answer=EMPTY_WORKSPACE_RESPONSE if workspace_empty else ABSTENTION_RESPONSE,
            confidence_score=0.0, generation_mode="abstained", requested_count=num_questions,
        )
    kinds = BLUEPRINTS[difficulty]
    questions, seen = [], []
    for start in range(0, num_questions, BATCH_SIZE):
        blueprint = {index + 1: kinds[index % len(kinds)]
                     for index in range(start, min(start + BATCH_SIZE, num_questions))}
        proposal = generate_study_json(
            query, evidences,
            f"Draft a DSA quiz at {difficulty} difficulty. Slots and exact types: {json.dumps(blueprint)}. "
            "Easy tests recognition and short examples; Medium applies rules or traces a short input; "
            "Hard requires multi-step reasoning, invariants, or edge cases. Cover distinct subtopics. "
            "Return a JSON array. Each object has slot, question, options (exactly four plausible strings), "
            "correct_index (0-3), explanation, sources (one to three context labels), question_type, difficulty, "
            "and topic (a short grounded subtopic). There must be exactly one correct answer. "
            "State assumptions and operation counts explicitly for traces/complexity. "
            "Skip any slot not supported by the evidence. Do not create fill-in-the-blank questions.",
            max_tokens=2400,
        )
        if proposal is None:
            break
        candidates = _candidates(proposal, blueprint, difficulty, len(evidences))
        if not candidates:
            continue
        # The checker must solve each question without seeing the proposed answer.
        review_input = [candidate.model_dump(exclude={"correct_index"}) for candidate in candidates]
        review = generate_study_json(
            json.dumps(review_input, ensure_ascii=False), evidences,
            f"Independently solve each candidate using only its cited context labels for the requested topic "
            f"{json.dumps(query, ensure_ascii=False)}. Reject ambiguity, "
            "off-topic questions, invalid distractors, unstated assumptions, or an unsupported explanation. "
            "Return a JSON array with slot, correct_index (0-3 you independently determined), "
            "unique_answer (true only if all other choices are wrong), explanation_supported (boolean), "
            "source (one of its cited labels), and quote (verbatim supporting excerpt from that source). "
            "Omit a candidate if the evidence cannot establish its answer.",
            max_tokens=1800,
        )
        if review is None:
            break
        for question in _verified(review, candidates, evidences):
            fingerprint = normalized_text(question.question)
            if not fingerprint or any(SequenceMatcher(None, fingerprint, prior).ratio() >= .88 for prior in seen):
                continue
            seen.append(fingerprint)
            questions.append(question)

    generated = len(questions)
    if not generated:
        return QuizResponse(
            query=query, answer="Chưa tạo được câu hỏi đã kiểm chứng. Hãy kiểm tra FIT-AI hoặc chọn chủ đề có thêm tài liệu.",
            confidence_score=0.0, generation_mode="unavailable", requested_count=num_questions,
        )
    answer = f"Đã tạo {generated}/{num_questions} câu hỏi đạt kiểm tra."
    fields = grounded_fields(query, answer, evidences, "hybrid_rerank", "llm",
                             [label for question in questions for label in question.sources])
    return QuizResponse(**fields, questions=questions, requested_count=num_questions, generated_count=generated)
