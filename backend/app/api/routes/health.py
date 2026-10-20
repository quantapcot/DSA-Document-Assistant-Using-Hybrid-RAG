# dsa_ai_agent/backend/app/api/routes/health.py
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/health", tags=["System Health"])
async def health_check():
    """Endpoint kiểm tra trạng thái hoạt động của Server."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "environment": settings.ENV,
        "version": "1.0.0"
    }