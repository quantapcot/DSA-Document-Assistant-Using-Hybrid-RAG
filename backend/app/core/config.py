# dsa_ai_agent/backend/app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "DSA Study Assistant AI Agent"
    ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Biến môi trường cho LLM & VectorDB (sử dụng ở các tuần sau)
    OPENAI_API_KEY: Optional[str] = None
    QDRANT_URL: str = "http://localhost:6333"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()