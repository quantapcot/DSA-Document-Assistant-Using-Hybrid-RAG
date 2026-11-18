"""Application settings — đọc từ backend/.env qua pydantic-settings."""
from __future__ import annotations

from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "DSA Study Assistant AI Agent"
    ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Vector DB ────────────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"

    # ── LLM Providers (C2) ──────────────────────────────────────────────────
    # Provider ưu tiên: "gemini" | "openai"
    LLM_PROVIDER: str = "gemini"
    # Tên model cụ thể (để trống = dùng default của provider)
    LLM_MODEL: str = ""
    # Google Gemini
    GEMINI_API_KEY: Optional[str] = None
    # OpenAI-compatible
    OPENAI_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
