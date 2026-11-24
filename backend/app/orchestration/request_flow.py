"""A1: connect intent router, retrieval, and generation (C2 integrated)."""
from __future__ import annotations

import logging
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from time import monotonic
from typing import Any

from anyio import to_thread
from fastapi import HTTPException

from app.core.config import settings
from app.core.telemetry import study_metrics
from app.generation.llm_client import ERROR_MESSAGES, request_deadline
from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.vector_store import scope_filter
from app.schemas.contracts import Citation
from .chat_retrieval import retrieve_chat_evidence
from .intent_router import plan_chat_query
from .study_evidence import curriculum_inventory, diverse_evidence, quiz_topic_filter
from .study_sessions import resolve_scope

logger = logging.getLogger("app.orchestration.request_flow")


@asynccontextmanager
async def study_request_budget(task):
    seconds = settings.request_timeout(task)
    token = request_deadline.set(monotonic() + seconds)
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError:
        raise HTTPException(504, {"code": "timeout", "message": ERROR_MESSAGES["timeout"]}) from None
    finally:
        request_deadline.reset(token)


async def run_in_threadpool(fn, *args, **kwargs):
    # The HTTP deadline can release the caller. An in-flight SDK call keeps its
    # own remaining-time limit, and later attempts see the expired deadline.
    return await to_thread.run_sync(partial(fn, *args, **kwargs), abandon_on_cancel=True)


def _scope_is_empty(engine, corpus_id, workspace_id):
    client = getattr(engine, "client", None)
    config = getattr(engine, "config", None)
    if client is None or config is None:
        return False
    if not client.collection_exists(config.collection):
        return True
    points, _ = client.scroll(
        config.collection, scroll_filter=scope_filter(corpus_id, workspace_id),
        limit=1, with_payload=False, with_vectors=False,
    )
    return not points


def _study_scope(body, request):
    workspace, corpus = resolve_scope(request, body.workspace_id, body.corpus_id)
    engine = getattr(request.app.state, "retrieval_engine", None)
    if engine is None:
        raise HTTPException(503, "Hệ thống truy xuất chưa sẵn sàng.")
    return engine, workspace, corpus


async def retrieve_for_study(body, request):
    query = " ".join(body.query.split())
    if not query:
        raise HTTPException(400, "Câu hỏi hoặc chủ đề không được để trống.")
    engine, workspace, corpus = _study_scope(body, request)
    plan = plan_chat_query(query)

    async def retrieve(search):
        return await execute_retrieval(engine, search, workspace_id=workspace, corpus_id=corpus)

    try:
        coverage = await retrieve_chat_evidence(plan, retrieve, workspace, corpus)
        if coverage.reason == "no_results" and await run_in_threadpool(_scope_is_empty, engine, corpus, workspace):
            coverage = replace(coverage, workspace_empty=True)
    except Exception:
        logger.error("Study retrieval failed")
        raise HTTPException(500, "Lỗi truy xuất tài liệu.") from None
    return plan, coverage


async def retrieve_for_quiz(body, request):
    """Use the plain topic first and one additional search only when evidence is thin."""
    query = body.topic
    engine, workspace, corpus = _study_scope(body, request)
    try:
        if await run_in_threadpool(_scope_is_empty, engine, corpus, workspace):
            return query, [], True
        search_topic, matches = quiz_topic_filter(query)

        async def matching(search):
            evidence, method = await execute_retrieval(
                engine, search, workspace_id=workspace, corpus_id=corpus, top_k=8,
            )
            return [{**item, "retrieval_method": method} for item in evidence if matches(item)]

        primary = await matching(search_topic)
        scope = {"workspace": workspace, "corpus": corpus, "apply_threshold": False}
        diverse = diverse_evidence([primary], **scope)
        if diverse and len(diverse) < min(body.num_questions, 3):
            extra = await matching(f"{search_topic} — nguyên lý và ví dụ / principles and examples")
            diverse = diverse_evidence([primary, extra], **scope)
        return query, diverse, False
    except Exception:
        logger.exception("Study evidence gathering failed")
        raise HTTPException(500, "Lỗi truy xuất tài liệu.") from None


async def inventory_for_roadmap(body, request):
    engine, workspace, corpus = _study_scope(body, request)
    started = monotonic()
    try:
        return await run_in_threadpool(curriculum_inventory, engine, workspace, corpus, body.topic)
    except Exception:
        logger.error("Roadmap inventory failed")
        raise HTTPException(500, "Lỗi đọc tài liệu trong phiên.") from None
    finally:
        if metrics := study_metrics.get():
            metrics.inventory_reads += 1
            metrics.retrieval_ms += (monotonic() - started) * 1000


async def execute_retrieval(
    engine: RetrievalEngine,
    query: str,
    *,
    workspace_id: str,
    corpus_id: str,
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """Truy xuất bằng chứng bất đồng bộ (không block event loop).

    Thứ tự ưu tiên:
    1. ``hybrid_rerank`` — BM25 + Dense + RRF + BGE Cross-Encoder.
    2. Fallback ``hybrid_rrf`` khi gặp lỗi CUDA / OOM từ reranker.
    3. Ném ngoại lệ trực tiếp nếu lỗi database / cấu hình scope.
    """
    method_used = "hybrid_rerank"

    async def measured(fn):
        started = monotonic()
        try:
            return await run_in_threadpool(fn)
        finally:
            if metrics := study_metrics.get():
                metrics.retrieval_calls += 1
                metrics.retrieval_ms += (monotonic() - started) * 1000

    def _retrieve_sync() -> list[dict[str, Any]]:
        return engine.retrieve(
            query=query,
            method="hybrid_rerank",
            top_k=top_k,
            corpus_id=corpus_id,
            workspace_id=workspace_id,
        )

    def _retrieve_fallback() -> list[dict[str, Any]]:
        return engine.retrieve(
            query=query,
            method="hybrid_rrf",
            top_k=top_k,
            corpus_id=corpus_id,
            workspace_id=workspace_id,
        )

    try:
        evidences = await measured(_retrieve_sync)
    except Exception as exc:
        err_str = str(exc).lower()
        if any(term in err_str for term in ["cuda", "out of memory", "oom", "rerank", "torch", "device"]):
            logger.warning("Reranking failed. Trying RRF in the same scope.")
            method_used = "hybrid_rrf"
            evidences = await measured(_retrieve_fallback)
        else:
            raise

    return evidences, method_used


def build_citations(evidences: list[dict[str, Any]]) -> list[Citation]:
    """Chuyển đổi danh sách evidence chunks thành đối tượng Citation."""
    from app.generation.citations import evidences_to_citations
    return evidences_to_citations(evidences)
