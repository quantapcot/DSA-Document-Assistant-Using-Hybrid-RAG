"""Generate and verify grounded learning steps for a requested topic."""
from dataclasses import asdict
import json
import logging
import re
from time import monotonic

from pydantic import BaseModel, ConfigDict, Field

from app.core.telemetry import study_metrics
from app.orchestration.study_evidence import normalized_text
from app.schemas.contracts import RoadmapContent, RoadmapItem, RoadmapResponse

from .abstention import EMPTY_WORKSPACE_RESPONSE
from .llm_client import ERROR_MESSAGES, LLMError
from .rag_engine import generate_study_json, grounded_fields
from .study_output import normalize_row, reject, rows, supports_quote, validate_row
from .study_planner import build_curriculum, curriculum_dependencies, order_topics, roadmap_batches

logger = logging.getLogger(__name__)
_RECOVERABLE = {"truncated_json", "invalid_json"}
_SOURCE_LABEL = re.compile(r"\[\d+\]")


class RoadmapDraft(RoadmapContent):
    model_config = ConfigDict(extra="forbid")
    topic_id: str = Field(min_length=1)


class RoadmapReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic_id: str = Field(min_length=1)
    on_topic: bool = Field(strict=True)
    supported: bool = Field(strict=True)


def _request_rows(batch, request, stage, identify):
    def unpack(payload):
        # The shared JSON reader may recover an intact row from a damaged array.
        if isinstance(payload, dict) and "topic_id" in payload:
            payload = [payload]
        entries = rows(payload, stage)
        if not entries and payload not in ([], {"items": []}):
            raise LLMError("invalid_json")
        return entries

    result, pending = [], batch
    try:
        payload = request(batch, 0.0)
        result = unpack(payload)
        if not isinstance(payload, dict) or "topic_id" not in payload:
            return result, None
        pending = [item for item in batch if identify(item) != payload["topic_id"]]
    except LLMError as error:
        if error.code not in _RECOVERABLE:
            return [], error.code
        reject(stage, error.code)

    error_code = None
    for item in pending:
        if metrics := study_metrics.get():
            metrics.content_retries += 1
        try:
            result.extend(unpack(request([item], 0.2)))
        except LLMError as error:
            error_code = error.code
            reject(stage, error_code)
            if error_code not in _RECOVERABLE:
                break
    return result, error_code


def _draft_batch(topic, batch, passages):
    def request(group, temperature):
        evidence = {label: passages[label - 1] for entry in group for label in entry.sources}
        instructions = (
            "Write one short Vietnamese learning objective and 1–5 concise key_knowledge bullets per "
            "assigned subtopic, retaining English DSA terms. Include only facts the cited passages support. "
            "Stay directly relevant to the requested topic. Skip a subtopic if its assigned evidence does "
            "not teach it. Use the assigned id as topic_id. Copy a short supporting quote in the original "
            "source language. Keep source labels out of objective and key_knowledge. "
            f"Assigned topics: {json.dumps([asdict(entry) for entry in group], ensure_ascii=False)}."
        )
        return generate_study_json(
            topic, evidence, instructions, task="revision", stage="roadmap.draft",
            max_tokens=900 + 240 * len(group), row_model=RoadmapDraft, temperature=temperature,
        )

    return _request_rows(batch, request, "roadmap.draft", lambda entry: entry.id)


def _valid_drafts(raw_rows, batch, passages):
    allowed = {entry.id: set(entry.sources) for entry in batch}
    accepted, seen = {}, set()
    for raw in raw_rows:
        draft = validate_row(RoadmapDraft, normalize_row(raw), "roadmap.content")
        if draft is None:
            continue
        if draft.topic_id in seen:
            reject("roadmap.content", "duplicate_topic")
            accepted.pop(draft.topic_id, None)
            continue
        seen.add(draft.topic_id)
        sources = set(draft.sources)
        text = (draft.objective, *draft.key_knowledge)
        if len(sources) != len(draft.sources) or not sources <= allowed.get(draft.topic_id, set()):
            reject("roadmap.citation", "invalid_topic_sources")
        elif not any(supports_quote(draft.support_quote, passages[label - 1]["text"]) for label in sources):
            reject("roadmap.quote", "unsupported_excerpt")
        elif not all(normalized_text(line) and not _SOURCE_LABEL.search(line) for line in text):
            reject("roadmap.content", "invalid_learning_text")
        else:
            accepted[draft.topic_id] = draft
    return accepted


def _review_drafts(topic, drafts, batch, passages):
    titles = {entry.id: entry.title for entry in batch}

    def request(group, temperature):
        evidence = {label: passages[label - 1] for draft in group for label in draft.sources}
        claims = [{"title": titles[draft.topic_id], **draft.model_dump(exclude={"support_quote"})}
                  for draft in group]
        instructions = (
            "Audit each roadmap step against the requested topic. Its title must be a concise concept name. "
            "Set on_topic to true only when the title, objective and every bullet directly teach the requested "
            "topic or a supported subtopic. Set supported to "
            "true only when the objective and every bullet are warranted by that step's cited passages "
            "and written in Vietnamese (English DSA terms are allowed). A merely related quote is not enough. "
            "Use only the supplied passages. Return topic_id, on_topic and supported for each audited step. "
            f"Steps to audit: {json.dumps(claims, ensure_ascii=False)}."
        )
        return generate_study_json(
            topic, evidence, instructions, task="revision", stage="roadmap.review",
            max_tokens=500 + 120 * len(group), row_model=RoadmapReview, temperature=temperature,
        )

    raw_reviews, error_code = _request_rows(list(drafts.values()), request, "roadmap.review", lambda draft: draft.topic_id)
    approved, seen = set(), set()
    for raw in raw_reviews:
        review = validate_row(RoadmapReview, normalize_row(raw, booleans=("on_topic", "supported")), "roadmap.review")
        if review is None or review.topic_id not in drafts:
            continue
        if review.topic_id not in seen and review.on_topic and review.supported:
            approved.add(review.topic_id)
        else:
            approved.discard(review.topic_id)
            reject("roadmap.review", "duplicate_review" if review.topic_id in seen else "off_topic_or_unsupported")
        seen.add(review.topic_id)
    return {identifier: drafts[identifier] for identifier in approved}, error_code


def generate_roadmap(topic, evidence, *, has_documents=False):
    no_match = f"Không tìm thấy đoạn tài liệu phù hợp để lập lộ trình cho chủ đề “{topic}” trong phiên này."
    if not evidence:
        answer = no_match if has_documents else EMPTY_WORKSPACE_RESPONSE
        return RoadmapResponse(query=topic, answer=answer, confidence_score=0.0, generation_mode="abstained")
    started = monotonic()
    catalog, passages = build_curriculum(evidence, topic)
    if not catalog:
        return RoadmapResponse(query=topic, answer=no_match,
                               confidence_score=0.0, generation_mode="abstained")
    graph = curriculum_dependencies(catalog, passages)
    ordered = order_topics(catalog, graph)
    contents, batches, error_code = {}, 0, None
    for batch in roadmap_batches(ordered, passages):
        batches += 1
        raw, draft_error = _draft_batch(topic, batch, passages)
        drafts = _valid_drafts(raw, batch, passages)
        review_error = None
        if drafts:
            approved, review_error = _review_drafts(topic, drafts, batch, passages)
            contents.update(approved)
        failure = review_error or draft_error
        if failure:
            error_code = failure
            if failure not in _RECOVERABLE:
                break

    completed, fingerprints, items, skipped = set(), set(), [], []
    for entry in ordered:
        draft = contents.get(entry.id)
        if draft is None:
            skipped.append(entry.title)
            continue
        fingerprint = normalized_text(" ".join((draft.objective, *draft.key_knowledge)))
        if fingerprint in fingerprints:
            reject("roadmap.assembly", "repeated_learning_content")
            skipped.append(entry.title)
            continue
        fingerprints.add(fingerprint)
        prerequisites = [parent for parent in graph[entry.id] if parent in completed]
        items.append(RoadmapItem(id=entry.id, title=entry.title, prerequisites=prerequisites,
                                 **draft.model_dump(exclude={"topic_id"})))
        completed.add(entry.id)

    logger.info("roadmap stage=final batches=%d accepted=%d topics=%d elapsed_ms=%.1f", batches, len(items),
                len(catalog), (monotonic() - started) * 1000)
    if not items:
        code = error_code or "invalid_output"
        return RoadmapResponse(query=topic, answer=ERROR_MESSAGES[code], confidence_score=0.0,
                               generation_mode="unavailable", generation_error=code, uncovered_topics=skipped)
    warnings = []
    if skipped:
        warnings.append("Một số mục chưa có nội dung được xác minh hoặc vượt giới hạn nguồn.")
    if error_code:
        warnings.append(ERROR_MESSAGES[error_code])
    labels = [label for item in items for label in item.sources]
    fields = grounded_fields(topic, f"Lộ trình ôn tập: {topic}", passages, "hybrid_rerank", "llm", labels)
    return RoadmapResponse(**fields, items=items, uncovered_topics=skipped, warnings=warnings,
                           generation_error=error_code)
