# ============================================================
# Dockerfile — DSA Assistant Backend (FastAPI + Uvicorn)
# ============================================================
FROM python:3.11-slim

# Metadata
LABEL maintainer="Team A — Track C DSA Assistant"
LABEL description="FastAPI backend: Hybrid RAG (BM25 + Qdrant Dense + BGE Reranker)"

# Biến môi trường hệ thống
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Cài dependencies trước (tận dụng Docker layer cache) ──────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Copy source code ────────────────────────────────────────────────────────
COPY . .

# ── Tạo thư mục dữ liệu runtime ─────────────────────────────────────────────
RUN mkdir -p /app/data/runtime/uploads \
             /app/data/runtime/qdrant_local

# ── Expose port ─────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Khởi động ───────────────────────────────────────────────────────────────
# Single-worker để tránh xung đột file lock Qdrant local.
# Khi dùng Qdrant server mode, có thể tăng lên --workers 4.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
