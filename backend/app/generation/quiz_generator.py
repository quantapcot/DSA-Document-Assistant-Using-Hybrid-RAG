"""Generate validated multiple-choice exercises from scoped evidence."""
import re

from app.schemas.contracts import QuizQuestion, QuizResponse

from .citations import MAX_EVIDENCES
from .rag_engine import abstention_fields, grounded_fields, generate_study_json

_WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def _extract_questions(evidence, limit):
    questions = []
    for label, item in enumerate(evidence, 1):
        excerpt = " ".join(item["text"].split())[:260]
        words = list(_WORD.finditer(excerpt))
        vocabulary = {match[0].casefold(): match[0] for match in words}
        if len(vocabulary) < 4:
            continue
        candidates = {}
        for match in words:
            candidates.setdefault(match[0].casefold(), match)
        for match in list(candidates.values())[:3]:
            answer = match[0]
            distractors = [word for key, word in vocabulary.items() if key != answer.casefold()][:3]
            index = len(questions) % 4
            distractors.insert(index, answer)
            questions.append(QuizQuestion(
                question="Điền từ còn thiếu trong đoạn trích: " + excerpt[:match.start()] + "____" + excerpt[match.end():],
                options=distractors, correct_index=index,
                explanation=f'Tài liệu dùng từ "{answer}" trong đoạn trích này [{label}].',
                sources=[label],
            ))
            if len(questions) == limit:
                return questions
    return questions


def generate_quiz(query, evidences, *, method_used="hybrid_rerank", num_questions=5,
                  workspace_empty=False):
    refusal = abstention_fields(query, evidences, method_used, workspace_empty=workspace_empty)
    if refusal:
        return QuizResponse(**refusal)
    evidence = evidences[:MAX_EVIDENCES]
    payload = generate_study_json(
        query, evidence,
        f"Return only a JSON array of up to {num_questions} objects. Each has question, "
        "options (four distinct strings), correct_index (0-3), explanation, and sources "
        "(a nonempty list of integer source labels).",
    )
    questions = []
    if isinstance(payload, list):
        try:
            questions = [QuizQuestion.model_validate(item) for item in payload[:num_questions]]
            if any(
                len({option.strip().casefold() for option in q.options if option.strip()}) != 4
                or not all(1 <= label <= len(evidence) for label in q.sources)
                for q in questions
            ):
                questions = []
        except ValueError:
            questions = []
    mode = "llm" if questions else "extractive"
    if not questions:
        questions = _extract_questions(evidence, num_questions)
    labels = [label for question in questions for label in question.sources]
    text = " ".join(question.explanation for question in questions)
    if not questions:
        text = "Tài liệu chưa đủ nội dung để tạo câu hỏi có bốn lựa chọn."
    fields = grounded_fields(query, text, evidence, method_used, mode, labels)
    return QuizResponse(**fields, questions=questions)
