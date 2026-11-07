"""B1: preserve document and page provenance — metadata factory và DocumentChunk builder."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from app.schemas.contracts import DocumentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_document_id(filename: str) -> str:
    """Tạo document_id từ tên file (không extension, lowercase, dấu cách → gạch dưới)."""
    stem = Path(filename).stem
    slug = re.sub(r"[^\w]+", "_", stem.lower()).strip("_")
    return slug or "unknown"


def _make_chunk_id(document_id: str, page: int, chunk_index: int, text: str) -> str:
    """Tạo chunk_id duy nhất: SHA-1 rút gọn của (doc_id + page + index + prefix)."""
    raw = f"{document_id}:p{page}:i{chunk_index}:{text[:64]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_chunk(
    *,
    filename: str,
    page: int,
    text: str,
    chunk_index: int,
    section: str | None = None,
    language: str = "unknown",
    extra_metadata: dict[str, Any] | None = None,
) -> DocumentChunk:
    """
    Tạo một DocumentChunk hoàn chỉnh từ đoạn văn bản đã làm sạch.

    Parameters
    ----------
    filename     : Tên file gốc (ví dụ: "001_textbook_CLRS.pdf").
    page         : Số trang trong tài liệu gốc (1-indexed).
    text         : Nội dung văn bản đã qua text_cleaner.
    chunk_index  : Vị trí thứ tự của chunk trong tài liệu (0-indexed).
    section      : Tiêu đề mục gần nhất (H1/H2) nếu detect được.
    language     : Mã ngôn ngữ ("vi", "en", "unknown").
    extra_metadata: Bất kỳ metadata bổ sung nào (token_count, file_type …).
    """
    document_id = _make_document_id(filename)
    chunk_id = _make_chunk_id(document_id, page, chunk_index, text)

    metadata: dict[str, Any] = {
        "language": language,
        "chunk_index": chunk_index,
        "char_count": len(text),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        page=page,
        section=section,
        text=text,
        metadata=metadata,
    )

