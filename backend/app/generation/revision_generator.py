"""Generate a sourced study calendar and reject invalid provider output."""
from datetime import datetime, timedelta, timezone

from app.schemas.contracts import RevisionResponse

from .abstention import ABSTENTION_RESPONSE, EMPTY_WORKSPACE_RESPONSE
from .rag_engine import generate_study_json, grounded_fields
from .study_planner import build_curriculum, plan_instructions, schedule_topics, validate_plan

_STUDY_TIMEZONE = timezone(timedelta(hours=7))


def resolve_exam_date(body, *, today=None):
    today = today or datetime.now(_STUDY_TIMEZONE).date()
    exam_date = body.exam_date or today + timedelta(days=body.days_remaining)
    if not 1 <= (exam_date - today).days <= 90:
        raise ValueError("Ngày thi phải cách hôm nay từ 1 đến 90 ngày.")
    return today, exam_date


def generate_revision(query, evidence, *, today, exam_date, budget, level, workspace_empty=False):
    if not evidence:
        return RevisionResponse(
            query=query, answer=EMPTY_WORKSPACE_RESPONSE if workspace_empty else ABSTENTION_RESPONSE,
            confidence_score=0.0, generation_mode="abstained",
        )
    catalog, passages = build_curriculum(evidence)
    chosen, slots = schedule_topics(catalog, today=today, exam_date=exam_date, budget=budget, level=level)
    uncovered = [topic.title for topic in catalog[len(chosen):]]
    warnings = []
    if uncovered:
        warnings.append(
            f"Kế hoạch ưu tiên {len(chosen)}/{len(catalog)} mục theo thứ tự trong tài liệu. "
            "Tăng thời gian hoặc thu hẹp chủ đề để học những mục còn lại."
        )
    if not any(kind == "mock_test" for _, kind, _, _ in slots):
        warnings.append("Thời gian hiện tại chỉ đủ cho phần học ưu tiên, chưa đủ phiên ôn lại và kiểm tra.")
    selected = passages[:chosen[-1].sources[-1]]
    settings = dict(exam_date=exam_date, budget=budget, level=level)
    request = plan_instructions(chosen, slots, **settings)
    payload = generate_study_json(query or "Ôn tất cả tài liệu", selected, request, max_tokens=6000)
    plan = validate_plan(payload, chosen, slots, selected, today=today, **settings)
    if plan is None:
        return RevisionResponse(
            query=query, answer="Chưa tạo được kế hoạch hợp lệ. Hãy thử lại hoặc kiểm tra kết nối FIT-AI.",
            confidence_score=0.0, generation_mode="unavailable",
        )
    labels = [label for day in plan.days for task in day.tasks for label in task.sources]
    answer = f"Kế hoạch gồm {len(plan.days)} ngày học, bao phủ {len(chosen)}/{len(catalog)} mục trong tài liệu."
    fields = grounded_fields(query, answer, selected, "hybrid_rerank", "llm", labels)
    return RevisionResponse(**fields, plan=plan, uncovered_topics=uncovered, warnings=warnings)
