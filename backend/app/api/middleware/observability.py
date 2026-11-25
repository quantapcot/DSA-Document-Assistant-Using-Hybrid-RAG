# dsa_ai_agent/backend/app/api/middleware/observability.py
import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.telemetry import StudyMetrics, study_metrics

logger = logging.getLogger("observability")
logger.setLevel(logging.INFO)

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware ghi log thời gian phản hồi (Latency) để phục vụ Team D đo lường."""
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        start_time = time.perf_counter()
        metrics = StudyMetrics()
        token = study_metrics.set(metrics)
        endpoints = {"/api/chat", "/api/quiz", "/api/revision", "/api/roadmap"}
        try:
            response = await call_next(request)
            response.headers["X-Process-Time-MS"] = f"{(time.perf_counter() - start_time) * 1000:.2f}"
            return response
        finally:
            study_metrics.reset(token)
            if request.url.path in endpoints:
                logger.info("study endpoint=%s elapsed_ms=%.1f retrieval_calls=%d inventory_reads=%d retrieval_ms=%.1f "
                            "provider_calls=%d provider_retries=%d content_retries=%d llm_ms=%s",
                            request.url.path, (time.perf_counter() - start_time) * 1000,
                            metrics.retrieval_calls, metrics.inventory_reads, metrics.retrieval_ms,
                            metrics.provider_calls, metrics.provider_retries, metrics.content_retries,
                            {task: round(ms, 1) for task, ms in metrics.llm_ms.items()})
