"""A1: connect intent router, retrieval, and generation (C2 integrated)."""
from __future__ import annotations

import logging
import asyncio
from contextlib import asynccontextmanager
from functools import partial
from time import monotonic
from typing import Any

from anyio import to_thread
from fastapi import HTTPException

from app.core.config import settings
from app.generation.llm_client import ERROR_MESSAGES, request_deadline
from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.vector_store import scope_filter
from app.schemas.contracts import Citation
from .study_evidence import (
    QUIZ_FACETS, curriculum_inventory, diverse_evidence, expanded_query, mentions_topic, topic_terms,
)

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
    engine = getattr(request.app.state, "retrieval_engine", None)
    if engine is None:
        raise HTTPException(503, "Hệ thống truy xuất chưa sẵn sàng.")
    workspace = (body.workspace_id or request.headers.get("X-Workspace-Id") or "default-workspace").strip()
    corpus = (body.corpus_id or request.headers.get("X-Corpus-Id") or "user-uploads").strip()
    if not workspace or not corpus:
        raise HTTPException(400, "Workspace và corpus không được để trống.")
    return engine, workspace, corpus


async def retrieve_for_study(body, request):
    query = " ".join(body.query.split())
    if not query:
        raise HTTPException(400, "Câu hỏi hoặc chủ đề không được để trống.")
    engine, workspace, corpus = _study_scope(body, request)
    try:
        evidence, method = await execute_retrieval(
            engine, query, workspace_id=workspace, corpus_id=corpus,
        )
        empty = not evidence and await run_in_threadpool(_scope_is_empty, engine, corpus, workspace)
    except Exception:
        logger.error("Study retrieval failed")
        raise HTTPException(500, "Lỗi truy xuất tài liệu.") from None
    return query, evidence, method, empty


async def retrieve_for_exercise(body, request, *, revision=False):
    """Collect diverse quiz evidence or a curriculum inventory for planning."""
    query = " ".join(body.query.split())
    if not query and not (revision and body.all_topics):
        raise HTTPException(400, "Chủ đề không được để trống.")
    engine, workspace, corpus = _study_scope(body, request)
    try:
        empty = await run_in_threadpool(_scope_is_empty, engine, corpus, workspace)
        if empty:
            return query, [], True
        if revision and body.all_topics:
            inventory = await run_in_threadpool(curriculum_inventory, engine, workspace, corpus)
            return query, inventory, not inventory
        terms = topic_terms(query)
        batches = []
        search_topic = expanded_query(query)
        for index, search in enumerate((search_topic, *(f"{search_topic} — {facet}" for facet in QUIZ_FACETS))):
            evidence, method = await execute_retrieval(
                engine, search, workspace_id=workspace, corpus_id=corpus, top_k=8,
            )
            if index == 0:
                # Topic gate: the plain topic query must find relevant passages
                # that actually mention the topic, otherwise the LLM is never called.
                on_topic = [item for item in evidence if mentions_topic(item, terms)]
                # The topic must literally appear in the retrieved passages (directly or
                # via a known alias such as "đệ quy" -> "recursion"). Rerank scores are
                # not used here: they are unreliable across languages and short chunks.
                if not on_topic:
                    logger.info("Study topic not found in workspace documents")
                    return query, [], False
            # Facet queries add generic words; keep only passages about the topic.
            batches.append([{**item, "retrieval_method": method}
                            for item in evidence if mentions_topic(item, terms)])
        # Passages were already filtered by topic, so the score cutoff is skipped.
        diverse = diverse_evidence(batches, workspace=workspace, corpus=corpus, apply_threshold=False)
        if not diverse:
            return query, [], False
        if revision and diverse:
            diverse = await run_in_threadpool(curriculum_inventory, engine, workspace, corpus, diverse)
        return query, diverse, False
    except Exception:
        logger.exception("Study evidence gathering failed")
        raise HTTPException(500, "Lỗi truy xuất tài liệu.") from None


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
        evidences = await run_in_threadpool(_retrieve_sync)
    except Exception as exc:
        err_str = str(exc).lower()
        if any(term in err_str for term in ["cuda", "out of memory", "oom", "rerank", "torch", "device"]):
            logger.warning("Reranking failed. Trying RRF in the same scope.")
            method_used = "hybrid_rrf"
            evidences = await run_in_threadpool(_retrieve_fallback)
        else:
            raise

    return evidences, method_used


def build_citations(evidences: list[dict[str, Any]]) -> list[Citation]:
    """Chuyển đổi danh sách evidence chunks thành đối tượng Citation."""
    from app.generation.citations import evidences_to_citations
    return evidences_to_citations(evidences)
