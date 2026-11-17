"""Chat endpoint — Hybrid Retrieval Engine + C2 LLM Generation."""
import logging
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from app.orchestration.intent_router import route_intent
from app.orchestration.request_flow import build_citations, execute_retrieval
from app.generation.rag_engine import build_answer
from app.schemas.contracts import ChatRequest, RAGResponse

logger = logging.getLogger("app.api.routes.chat")

router = APIRouter()


@router.post("/api/chat", response_model=RAGResponse, tags=["Chat Agent"])
async def chat_endpoint(request_body: ChatRequest, request: Request):
    # ── Validate input ───────────────────────────────────────────────────────
    question = " ".join(request_body.query.strip().split())
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Câu hỏi không được để trống.",
        )

    # ── 1. Phân loại ý định ──────────────────────────────────────────────────
    intent = route_intent(question)
    logger.info("Xử lý câu hỏi: '%s' | Intent: %s", question, intent)

    # ── 2. Lấy RetrievalEngine Singleton ────────────────────────────────────
    engine = getattr(request.app.state, "retrieval_engine", None)
    if not engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hệ thống Retrieval Engine chưa sẵn sàng hoặc đang khởi động.",
        )

    # ── 3. Xác định scope (Workspace / Corpus) ────────────────────────────────
    workspace_id = (
        request_body.workspace_id
        or request.headers.get("X-Workspace-Id")
        or "default-workspace"
    )
    corpus_id = (
        request_body.corpus_id
        or request.headers.get("X-Corpus-Id")
        or "user-uploads"
    )

    # ── 4. Truy xuất evidence (async, không block event loop) ─────────────────
    evidences: list[dict] = []
    method_used = "hybrid_rerank"
    try:
        evidences, method_used = await execute_retrieval(
            engine,
            query=question,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            top_k=5,
        )

        # Fallback: thử corpus benchmark nếu workspace mặc định không có kết quả
        if not evidences and workspace_id == "default-workspace":
            logger.info(
                "Chưa có chunk trong 'default-workspace', thử 'research/benchmark-v1'..."
            )
            try:
                benchmark_evs, b_method = await execute_retrieval(
                    engine,
                    query=question,
                    workspace_id="research",
                    corpus_id="benchmark-v1",
                    top_k=5,
                )
                if benchmark_evs:
                    evidences = benchmark_evs
                    method_used = b_method
            except Exception as b_exc:
                logger.debug("Benchmark corpus check skipped: %s", b_exc)

    except Exception as exc:
        logger.error("Lỗi khi truy xuất dữ liệu: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi truy xuất tài liệu: {exc}",
        )

    # ── 5. C2 RAG Generation (LLM hoặc Extractive fallback) ──────────────────
    try:
        answer, citations, confidence = await run_in_threadpool(
            build_answer,
            question,
            evidences,
            intent,
            method_used=method_used,
        )
    except Exception as exc:
        logger.error("Lỗi khi sinh câu trả lời: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi sinh câu trả lời: {exc}",
        )

    return RAGResponse(
        query=question,
        answer=answer,
        citations=citations,
        confidence_score=confidence,
        is_safe=True,
    )
