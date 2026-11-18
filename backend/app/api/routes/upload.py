"""Upload and ingest PDF/DOCX documents into Hybrid RAG pipeline."""
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.orchestration.ingestion_flow import process_and_index_document

logger = logging.getLogger("app.api.routes.upload")

router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parents[4] / "data" / "runtime" / "uploads"
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB


@router.post("/api/upload", tags=["Upload & Ingestion"])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    workspace_id: str = Form("default-workspace"),
    corpus_id: str = Form("user-uploads"),
):
    # ── Validate filename & extension ────────────────────────────────────────
    filename = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Hệ thống chỉ hỗ trợ file định dạng .pdf hoặc .docx",
        )

    # ── Validate size ────────────────────────────────────────────────────────
    content = await file.read(MAX_FILE_SIZE + 1)
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File rỗng.",
        )
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Kích thước file vượt quá giới hạn tối đa 15 MB.",
        )

    # ── 1. Lưu file vật lý ──────────────────────────────────────────────────
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_stored_filename = f"{uuid4().hex}_{filename}"
    saved_path = UPLOAD_DIR / safe_stored_filename
    saved_path.write_bytes(content)

    # ── 2. Tiền xử lý (B1) & Lập chỉ mục (C1) — chạy trong thread pool ──────
    # Tránh block FastAPI event loop khi Embedding Model chạy CPU/GPU nặng.
    retrieval_config = getattr(request.app.state, "retrieval_config", None)
    retrieval_engine = getattr(request.app.state, "retrieval_engine", None)
    qdrant_client = getattr(retrieval_engine, "client", None) if retrieval_engine else None

    try:
        result = await run_in_threadpool(
            process_and_index_document,
            saved_path,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            config=retrieval_config,
            client=qdrant_client,
        )
        return {
            "status": "success",
            "filename": filename,
            "stored_as": safe_stored_filename,
            "size_kb": round(len(content) / 1024, 2),
            "total_chunks": result["total_chunks"],
            "indexed_chunks": result["indexed_chunks"],
            "scope_count": result["scope_count"],
            "workspace_id": result["workspace_id"],
            "corpus_id": result["corpus_id"],
            "message": (
                f"Tài liệu '{filename}' đã được bóc tách và lập chỉ mục thành công "
                f"({result['indexed_chunks']} chunks vào kho ngữ liệu)."
            ),
        }
    except Exception as exc:
        # Dọn dẹp file đã lưu nếu indexing thất bại để tránh rác trên ổ đĩa
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)
        logger.error("Lỗi khi xử lý & index file '%s': %s", filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi khi xử lý tài liệu: {exc}",
        )
