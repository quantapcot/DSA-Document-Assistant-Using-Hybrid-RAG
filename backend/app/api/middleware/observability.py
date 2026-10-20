# dsa_ai_agent/backend/app/api/middleware/observability.py
import time
import json
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("observability")
logger.setLevel(logging.INFO)

class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Middleware ghi log thời gian phản hồi (Latency) để phục vụ Team D đo lường."""
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Ghi log thời gian xử lý vào Response Header
        response.headers["X-Process-Time-MS"] = str(process_time_ms)
        return response