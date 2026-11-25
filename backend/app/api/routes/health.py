# dsa_ai_agent/backend/app/api/routes/health.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", tags=["System Health"])
async def health_check():
    """Endpoint kiểm tra trạng thái hoạt động của Server."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.ENV,
        "version": "1.0.0",
        "study_request_timeouts": {task: settings.request_timeout(task)
                                   for task in ("chat", "quiz", "revision")},
    }


@router.get("/ready", tags=["System Health"])
async def readiness_check(request: Request):
    """Return 503 when the Retrieval engine or its Qdrant backend is unavailable."""
    engine = getattr(request.app.state, "retrieval_engine", None)
    if engine is None or getattr(engine, "client", None) is None:
        raise HTTPException(503, detail="retrieval_unavailable")
    try:
        await run_in_threadpool(engine.client.get_collections)
    except Exception as exc:
        logger.warning("Qdrant readiness failed: %s", type(exc).__name__)
        raise HTTPException(503, detail="qdrant_unavailable") from None
    config = getattr(engine, "config", None)
    return {"status": "ready", "retrieval": "ready", "qdrant": "ready",
            "reranker_configured": bool(getattr(config, "reranker_model", None))}
