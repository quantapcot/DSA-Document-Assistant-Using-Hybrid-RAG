"""A1: connect intent router, retrieval, and generation (C2 integrated)."""
from __future__ import annotations

import logging
from typing import Any

from fastapi.concurrency import run_in_threadpool

from app.retrieval.hybrid_search import RetrievalEngine
from app.schemas.contracts import Citation

logger = logging.getLogger("app.orchestration.request_flow")


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
            logger.warning(
                "Cross-Encoder Reranker gặp sự cố (%s). Tự động fallback sang 'hybrid_rrf'", exc
            )
            method_used = "hybrid_rrf"
            evidences = await run_in_threadpool(_retrieve_fallback)
        else:
            logger.error("Lỗi truy xuất hệ thống (không thuộc lỗi model GPU): %s", exc)
            raise exc

    return evidences, method_used


def build_citations(evidences: list[dict[str, Any]]) -> list[Citation]:
    """Chuyển đổi danh sách evidence chunks thành đối tượng Citation."""
    from app.generation.citations import evidences_to_citations
    return evidences_to_citations(evidences)
