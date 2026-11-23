"""Generate and validate sourced learning activities for a Python-owned roadmap."""
from dataclasses import asdict
import json
import logging
from time import monotonic

from pydantic import BaseModel, ConfigDict, Field

from app.core.telemetry import study_metrics
from app.orchestration.study_evidence import normalized_text
from app.schemas.contracts import RoadmapItem, RoadmapResponse, SourceLabel

from .abstention import EMPTY_WORKSPACE_RESPONSE
from .llm_client import ERROR_MESSAGES, LLMError
from .rag_engine import generate_study_json, grounded_fields
from .study_output import normalize_row, reject, rows, supports_quote, validate_row
from .study_planner import build_curriculum, curriculum_dependencies, order_topics, roadmap_batches

logger = logging.getLogger(__name__)
_TITLE = "Kế hoạch học tập"


class RoadmapDraft(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    topic_id: str = Field(min_length=1)
    objective: str = Field(min_length=1, max_length=300)
    knowledge: list[str] = Field(min_length=1, max_length=5)
    activity: str = Field(min_length=1, max_length=400)
    sources: list[SourceLabel] = Field(min_length=1, max_length=2)
    support_quote: str = Field(min_length=20, max_length=500)


def _draft_batch(batch, passages):
    evidence = {label: passages[label - 1] for topic in batch for label in topic.sources}
    instructions = (
        "Write one concise Vietnamese learning activity per supported topic, retaining English DSA terms. "
        "Use the given topic_id and source labels. Include a learning objective, 1–3 key knowledge points, "
        "and a practical activity. Copy a short supporting quote in the source language. "
        "Use only facts supported by those sources. Skip topics with insufficient evidence. "
        "Python owns the titles and order. Do not add schedules or new topics. "
        f"Assigned topics: {json.dumps([asdict(topic) for topic in batch], ensure_ascii=False)}."
    )
    for attempt in range(2):
        if attempt and (metrics := study_metrics.get()):
            metrics.content_retries += 1
        try:
            return generate_study_json(
                _TITLE, evidence, instructions, task="revision", stage="roadmap.draft",
                max_tokens=900 + 240 * len(batch), row_model=RoadmapDraft,
                temperature=0.2 * attempt,
            )
        except LLMError as error:
            if attempt or error.code not in {"truncated_json", "invalid_json"}:
                raise
            reject("roadmap.draft", error.code)


def _valid_drafts(payload, batch, passages):
    allowed = {topic.id: set(topic.sources) for topic in batch}
    accepted, seen = {}, set()
    for raw in rows(payload, "roadmap.content"):
        draft = validate_row(RoadmapDraft, normalize_row(raw), "roadmap.content")
        if draft is None:
            continue
        if draft.topic_id in seen:
            reject("roadmap.content", "duplicate_topic")
            accepted.pop(draft.topic_id, None)
            continue
        seen.add(draft.topic_id)
        sources = set(draft.sources)
        if len(sources) != len(draft.sources) or not sources <= allowed.get(draft.topic_id, set()):
            reject("roadmap.citation", "invalid_topic_sources")
        elif not any(supports_quote(draft.support_quote, passages[label - 1]["text"]) for label in sources):
            reject("roadmap.quote", "unsupported_excerpt")
        elif not all(map(normalized_text, (draft.objective, draft.activity, *draft.knowledge))):
            reject("roadmap.content", "empty_learning_content")
        else:
            accepted[draft.topic_id] = draft
    return accepted


def generate_roadmap(evidence):
    if not evidence:
        return RoadmapResponse(query=_TITLE, answer=EMPTY_WORKSPACE_RESPONSE, confidence_score=0.0,
                               generation_mode="abstained")
    started = monotonic()
    catalog, passages = build_curriculum(evidence)
    if not catalog:
        return RoadmapResponse(query=_TITLE, answer="Chưa xác định được chủ đề có tên từ những tài liệu này.",
                               confidence_score=0.0, generation_mode="abstained")
    graph = curriculum_dependencies(catalog, passages)
    ordered = order_topics(catalog, graph)
    contents, calls, error_code = {}, 0, None
    for batch in roadmap_batches(ordered, passages):
        calls += 1
        try:
            contents.update(_valid_drafts(_draft_batch(batch, passages), batch, passages))
        except LLMError as error:
            error_code = error.code
            if error.code not in {"truncated_json", "invalid_json"}:
                break
    completed, fingerprints, items, skipped = set(), set(), [], []
    for topic in ordered:
        draft = contents.get(topic.id)
        if draft is None or not set(graph[topic.id]) <= completed:
            skipped.append(topic.title)
            continue
        fingerprint = normalized_text(" ".join((draft.objective, draft.activity, *draft.knowledge)))
        if fingerprint in fingerprints:
            reject("roadmap.assembly", "repeated_learning_content")
            skipped.append(topic.title)
            continue
        fingerprints.add(fingerprint)
        completed.add(topic.id)
        items.append(RoadmapItem(id=topic.id, title=topic.title, prerequisites=graph[topic.id],
                                 **draft.model_dump(exclude={"topic_id"})))

    logger.info("roadmap stage=final batches=%d accepted=%d topics=%d elapsed_ms=%.1f", calls, len(items),
                len(catalog), (monotonic() - started) * 1000)
    if not items:
        code = error_code or "invalid_output"
        return RoadmapResponse(query=_TITLE, answer=ERROR_MESSAGES[code], confidence_score=0.0,
                               generation_mode="unavailable", generation_error=code, uncovered_topics=skipped)
    warnings = []
    if skipped:
        warnings.append("Một số chủ đề chưa có nội dung được xác minh hoặc vượt giới hạn nguồn.")
    if error_code:
        warnings.append(ERROR_MESSAGES[error_code])
    labels = [label for item in items for label in item.sources]
    fields = grounded_fields(_TITLE, f"Lộ trình gồm {len(items)}/{len(catalog)} chủ đề trong tài liệu.",
                             passages, "hybrid_rerank", "llm", labels)
    return RoadmapResponse(**fields, items=items, uncovered_topics=skipped, warnings=warnings,
                           generation_error=error_code)
