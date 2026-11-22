"""Generate a sourced study calendar and reject invalid provider output."""
from datetime import datetime, timedelta, timezone

from app.schemas.contracts import RevisionResponse

from .abstention import ABSTENTION_RESPONSE, EMPTY_WORKSPACE_RESPONSE, topic_missing_response
from .llm_client import ERROR_MESSAGES, LLMError
from .rag_engine import generate_study_json, grounded_fields
from .study_planner import assemble_plan, batch_request, build_curriculum, schedule_topics, validate_batch
from .study_output import reject

_STUDY_TIMEZONE = timezone(timedelta(hours=7))
REVISION_BATCH_SIZE = 3
REVISION_BATCH_OUTPUT = 2300


def resolve_exam_date(body, *, today=None):
    today = today or datetime.now(_STUDY_TIMEZONE).date()
    exam_date = body.exam_date or today + timedelta(days=body.days_remaining)
    if not 1 <= (exam_date - today).days <= 90:
        raise ValueError("Ngày thi phải cách hôm nay từ 1 đến 90 ngày.")
    return today, exam_date


def generate_revision(query, evidence, *, today, exam_date, budget, level, workspace_empty=False):
    if not evidence:
        return RevisionResponse(
            query=query, answer=EMPTY_WORKSPACE_RESPONSE if workspace_empty
            else (topic_missing_response(query) if query else ABSTENTION_RESPONSE),
            confidence_score=0.0, generation_mode="abstained",
        )
    catalog, passages = build_curriculum(evidence)
    chosen, slots = schedule_topics(catalog, passages, today=today, exam_date=exam_date, budget=budget, level=level)
    chosen_ids = {topic.id for topic in chosen}
    uncovered = [topic.title for topic in catalog if topic.id not in chosen_ids]
    if not slots:
        return RevisionResponse(
            query=query, answer="Ngân sách hiện tại không đủ cho mục nào có nguồn tham khảo.",
            confidence_score=0.0, generation_mode="unavailable", uncovered_topics=uncovered,
        )
    warnings = []
    if uncovered:
        warnings.append(
            f"Kế hoạch xếp {len(chosen)}/{len(catalog)} mục theo tiên quyết, thời gian và giới hạn nguồn. "
            "Tăng thời gian hoặc thu hẹp chủ đề để học những mục còn lại."
        )
    if not any(slot.kind == "mock_test" for slot in slots):
        warnings.append("Thời gian hiện tại chỉ đủ cho phần học ưu tiên, chưa đủ phiên ôn lại và kiểm tra.")
    for kind, name in (("review", "Ôn lại"), ("mock_test", "Kiểm tra tổng hợp")):
        covered = {topic for slot in slots if slot.kind == kind for topic in slot.topic_ids}
        if covered and covered != chosen_ids:
            warnings.append(f"{name} bao phủ {len(covered)}/{len(chosen_ids)} mục trong thời lượng đã chọn.")
    parameters = dict(exam_date=exam_date, budget=budget, level=level)
    try:
        contents = {}
        for start in range(0, len(slots), REVISION_BATCH_SIZE):
            batch = slots[start:start + REVISION_BATCH_SIZE]
            contents.update(_generate_batch(query, batch, chosen, passages, level))
        if not contents:
            raise LLMError("invalid_output")
        planned = len(slots)
        chosen, slots = _prune_failed(chosen, slots, contents)
        dropped = planned - len(slots)
        contents = {slot.id: contents[slot.id] for slot in slots}
        if not slots:
            raise LLMError("invalid_output")
        if dropped:
            warnings.append(f"Bỏ {dropped} buổi học vì FIT-AI không tạo được nội dung có trích dẫn hợp lệ.")
        plan = assemble_plan(contents, chosen, slots, passages, today=today, **parameters)
        if plan is None:
            raise LLMError("invalid_output")
    except LLMError as error:
        return RevisionResponse(
            query=query, answer=ERROR_MESSAGES[error.code],
            confidence_score=0.0, generation_mode="unavailable", generation_error=error.code,
        )
    labels = [label for day in plan.days for task in day.tasks for label in task.sources]
    answer = f"Kế hoạch gồm {len(plan.days)} ngày học, bao phủ {len(chosen)}/{len(catalog)} mục trong tài liệu."
    fields = grounded_fields(query, answer, passages, "hybrid_rerank", "llm", labels)
    return RevisionResponse(**fields, plan=plan, uncovered_topics=uncovered, warnings=warnings)


def _generate_batch(query, slots, chosen, passages, level, *, retry_single=True):
    context, instructions = batch_request(slots, chosen, passages, level)
    try:
        payload = generate_study_json(
            query or "Ôn tất cả tài liệu", context, instructions,
            max_tokens=REVISION_BATCH_OUTPUT, task="revision", stage="revision.batch_json",
        )
        result = validate_batch(payload, slots, chosen, passages)
        if result is None:
            raise LLMError("invalid_output")
        return result
    except LLMError as error:
        reject("revision.batch", error.code, slot=slots[0].id if len(slots) == 1 else None)
        if error.code not in {"invalid_json", "truncated_json", "invalid_output"} or not retry_single:
            raise
        # A failed batch gets one smaller attempt per slot, with the exact same
        # scheduler IDs and citations. A slot that still fails is dropped and
        # the plan is pruned later, instead of discarding the whole plan.
        result = {}
        for slot in slots:
            try:
                result.update(_generate_batch(query, [slot], chosen, passages, level, retry_single=False))
            except LLMError as slot_error:
                if slot_error.code not in {"invalid_json", "truncated_json", "invalid_output"}:
                    raise
        return result


def _prune_failed(chosen, slots, contents):
    """Drop failed slots and everything that depends on a topic left unlearned."""
    learned, kept = set(), []
    for slot in slots:
        if slot.id not in contents:
            continue
        if slot.kind == "learn":
            if not set(slot.prerequisites) <= learned:
                continue
            learned.update(slot.topic_ids)
            kept.append(slot)
            continue
        topics = tuple(topic for topic in slot.topic_ids if topic in learned)
        if topics == slot.topic_ids:
            kept.append(slot)
    # Review slots reference topics by ID; drop any whose topics were removed.
    kept = [slot for slot in kept if slot.kind == "learn" or set(slot.topic_ids) <= learned]
    return [topic for topic in chosen if topic.id in learned], kept
