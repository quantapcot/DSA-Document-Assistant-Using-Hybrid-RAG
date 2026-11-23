"""Application settings — đọc từ backend/.env qua pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[2]


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
    GEMINI_API_KEY: SecretStr | None = None
    # OpenAI-compatible
    OPENAI_API_KEY: SecretStr | None = None
    LLM_BASE_URL: str | None = None
    LLM_CHAT_TIMEOUT: float = Field(30, gt=0, le=300)
    LLM_QUIZ_GENERATION_TIMEOUT: float = Field(90, gt=0, le=300)
    LLM_QUIZ_VERIFICATION_TIMEOUT: float = Field(40, gt=0, le=300)
    LLM_REVISION_TIMEOUT: float = Field(45, gt=0, le=300)
    CHAT_REQUEST_TIMEOUT: float = Field(90, gt=0, le=900)
    QUIZ_REQUEST_TIMEOUT: float = Field(240, gt=0, le=900)
    REVISION_REQUEST_TIMEOUT: float = Field(240, gt=0, le=900)
    REVISION_MAX_TOPICS: int = Field(20, ge=1, le=60)
    REVISION_SOURCE_TOKEN_BUDGET: int = Field(12000, ge=300, le=40000)

    def request_timeout(self, task: str) -> float:
        return getattr(self, f"{task.upper()}_REQUEST_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=(_BACKEND_DIR.parent / ".env", _BACKEND_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
