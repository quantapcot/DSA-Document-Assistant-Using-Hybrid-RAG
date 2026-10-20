# dsa_ai_agent/backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes import health, chat
from app.api.middleware.observability import ObservabilityMiddleware

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API Server cho Trợ lý Học tập môn DSA (Track C)",
    version="1.0.0"
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

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} API Server",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)