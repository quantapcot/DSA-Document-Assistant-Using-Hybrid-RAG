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

    GROQ_API_KEY: SecretStr | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    LLM_CHAT_TIMEOUT: float = Field(30, gt=0, le=300)
    LLM_QUIZ_GENERATION_TIMEOUT: float = Field(90, gt=0, le=300)
    LLM_QUIZ_VERIFICATION_TIMEOUT: float = Field(40, gt=0, le=300)
    LLM_REVISION_TIMEOUT: float = Field(45, gt=0, le=300)
    CHAT_REQUEST_TIMEOUT: float = Field(90, gt=0, le=900)
    QUIZ_REQUEST_TIMEOUT: float = Field(240, gt=0, le=900)
    REVISION_REQUEST_TIMEOUT: float = Field(240, gt=0, le=900)
    ROADMAP_MAX_TOPICS: int = Field(20, ge=1, le=60)
    ROADMAP_SOURCE_BUDGET: int = Field(14000, ge=1000, le=48000)
    SESSION_TTL_SECONDS: int = Field(8 * 3600, gt=0)
    SESSION_CLEANUP_SECONDS: int = Field(300, gt=0)
    SESSION_RUNTIME_DIR: Path = _BACKEND_DIR.parent / "data" / "runtime"
    EXPORT_DEBUG_CHUNKS: bool = False

    @property
    def upload_dir(self) -> Path:
        return Path(self.SESSION_RUNTIME_DIR) / "uploads"

    def request_timeout(self, task: str) -> float:
        return getattr(self, f"{task.upper()}_REQUEST_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=(_BACKEND_DIR.parent / ".env", _BACKEND_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
