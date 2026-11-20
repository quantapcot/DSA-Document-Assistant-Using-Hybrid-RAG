"""A1: connect intent router, retrieval, and generation (C2 integrated)."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool

from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.vector_store import scope_filter
from app.schemas.contracts import Citation

logger = logging.getLogger("app.orchestration.request_flow")


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


async def retrieve_for_study(body, request):
    query = " ".join(body.query.split())
    if not query:
        raise HTTPException(400, "Câu hỏi hoặc chủ đề không được để trống.")
    engine = getattr(request.app.state, "retrieval_engine", None)
    if engine is None:
        raise HTTPException(503, "Hệ thống truy xuất chưa sẵn sàng.")
    workspace = (body.workspace_id or request.headers.get("X-Workspace-Id") or "default-workspace").strip()
    corpus = (body.corpus_id or request.headers.get("X-Corpus-Id") or "user-uploads").strip()
    if not workspace or not corpus:
        raise HTTPException(400, "Workspace và corpus không được để trống.")
    try:
        evidence, method = await execute_retrieval(
            engine, query, workspace_id=workspace, corpus_id=corpus,
        )
        empty = not evidence and await run_in_threadpool(_scope_is_empty, engine, corpus, workspace)
    except Exception:
        logger.error("Study retrieval failed")
        raise HTTPException(500, "Lỗi truy xuất tài liệu.") from None
    return query, evidence, method, empty


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
