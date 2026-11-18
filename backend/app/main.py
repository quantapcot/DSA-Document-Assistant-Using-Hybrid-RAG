# dsa_ai_agent/backend/app/main.py
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import health, chat, upload, documents
from app.api.middleware.observability import ObservabilityMiddleware
from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine

logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo RetrievalEngine Singleton một lần duy nhất khi ứng dụng khởi động
    logger.info("Initializing Retrieval Engine Singleton...")
    config_path = Path(__file__).resolve().parents[2] / "configs" / "retrieval.mvp.yaml"
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[2] / "configs" / "retrieval.example.yaml"

    if config_path.exists():
        logger.info(f"Loading retrieval config from: {config_path}")
        retrieval_config = RetrievalConfig.from_yaml(config_path)
    else:
        logger.warning("Config YAML not found; loading retrieval config from env/defaults")
        retrieval_config = RetrievalConfig.from_env()

    app.state.retrieval_config = retrieval_config
    try:
        app.state.retrieval_engine = RetrievalEngine(config=retrieval_config)
        logger.info("Retrieval Engine singleton initialized successfully.")

        # Warmup models: Nạp sẵn weights vào RAM/GPU ngay khi server khởi động
        # để tránh bị timeout (cold-start) ở request đầu tiên của người dùng
        try:
            from app.retrieval.embeddings import _get_model as get_emb_model
            get_emb_model(retrieval_config.embedding_model, retrieval_config.embedding_device)

            if retrieval_config.reranker_model:
                from app.retrieval.reranker import _get_model as get_rerank_model
                get_rerank_model(
                    retrieval_config.reranker_model,
                    retrieval_config.reranker_device,
                    retrieval_config.reranker_max_length,
                )
            logger.info("Models preloaded (warmup) successfully.")
        except Exception as warm_exc:
            logger.warning("Model warmup notice: %s", warm_exc)

    except Exception as exc:
        logger.error(f"Failed to initialize RetrievalEngine at startup: {exc}")
        app.state.retrieval_engine = None

    yield

    logger.info("Application shutdown: cleaning up resources...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API Server cho Trợ lý Học tập môn DSA (Track C)",
    version="1.0.0",
    lifespan=lifespan,
)

# Cấu hình CORS cho phép Streamlit Frontend (Port 8501) kết nối sang
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tích hợp Observability Middleware
app.add_middleware(ObservabilityMiddleware)

# Đăng ký các Routers
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(documents.router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} API Server",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
