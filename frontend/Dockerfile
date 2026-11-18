# ============================================================
# Dockerfile — DSA Assistant Frontend (Streamlit)
# ============================================================
FROM python:3.11-slim

LABEL maintainer="Team A — Track C DSA Assistant"
LABEL description="Streamlit frontend: Chat UI, Upload UI, Quiz UI, Revision UI"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Dependencies ─────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Source code ──────────────────────────────────────────────────────────────
COPY . .

# ── Streamlit config ─────────────────────────────────────────────────────────
# File .streamlit/config.toml đã có sẵn trong repo — override qua env nếu cần
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
