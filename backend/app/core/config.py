"""Application settings — đọc từ backend/.env qua pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from pydantic import SecretStr
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

    model_config = SettingsConfigDict(
        env_file=(_BACKEND_DIR.parent / ".env", _BACKEND_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
