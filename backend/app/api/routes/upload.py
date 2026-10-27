"""PoC: validate and save an uploaded PDF, without processing its content."""
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parents[4] / "data" / "runtime" / "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB for this demo


@router.post("/api/upload", tags=["PoC Upload"])
async def upload_pdf(file: UploadFile = File(...)):
    filename = Path((file.filename or "").replace("\\", "/")).name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chỉ chấp nhận file .pdf.",
        )

    # Upload-only PoC: check basic PDF signature; no parsing or indexing.
    content = await file.read(MAX_FILE_SIZE + 1)
    if not content or not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File rỗng hoặc không có định dạng PDF hợp lệ.",
        )
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PoC chỉ hỗ trợ PDF tối đa 10 MB.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored_filename = f"{uuid4().hex}.pdf"  # Avoid path traversal and filename collisions
    (UPLOAD_DIR / stored_filename).write_bytes(content)

    return {
        "status": "success",
        "filename": filename,
        "size_kb": round(len(content) / 1024, 2),
        "message": f"Đã lưu '{filename}' thành công. Chưa xử lý nội dung PDF (PoC).",
    }