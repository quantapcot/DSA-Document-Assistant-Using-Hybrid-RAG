"""B1: document_parser — PDF/DOCX → list[PageText], điểm vào của pipeline xử lý tài liệu."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure trung gian (trước khi tạo DocumentChunk)
# ---------------------------------------------------------------------------

@dataclass
class PageText:
    """Văn bản thô của một trang/đoạn trong tài liệu."""
    page: int          # 1-indexed
    raw_text: str      # chưa làm sạch
    file_type: str     # "pdf" | "docx"


# ---------------------------------------------------------------------------
# PDF Parser (PyMuPDF)
# ---------------------------------------------------------------------------

def _parse_pdf(file_path: Path) -> list[PageText]:
    """
    Trích xuất text từ PDF theo từng trang bằng PyMuPDF.
    Dùng layout="text" để giữ thứ tự đọc tự nhiên.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Cài đặt PyMuPDF: pip install pymupdf")

    pages: list[PageText] = []
    with fitz.open(str(file_path)) as doc:
        for i, page in enumerate(doc):
            # get_text("text") giữ thứ tự đọc tự nhiên, trả về plain text
            text = page.get_text("text")
            if text.strip():
                pages.append(PageText(page=i + 1, raw_text=text, file_type="pdf"))
    return pages


# ---------------------------------------------------------------------------
# DOCX Parser (python-docx)
# ---------------------------------------------------------------------------

def _parse_docx(file_path: Path) -> list[PageText]:
    """
    Trích xuất text từ DOCX theo paragraph.
    DOCX không có khái niệm trang → tách theo heading H1 để tạo "trang ảo".
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("Cài đặt python-docx: pip install python-docx")

    doc = Document(str(file_path))

    pages: list[PageText] = []
    current_lines: list[str] = []
    current_page = 1

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Nếu gặp heading cấp 1 → tạo trang mới
        if para.style.name.startswith("Heading 1") and current_lines:
            pages.append(PageText(
                page=current_page,
                raw_text="\n".join(current_lines),
                file_type="docx",
            ))
            current_lines = []
            current_page += 1

        current_lines.append(text)

    # Flush phần còn lại
    if current_lines:
        pages.append(PageText(
            page=current_page,
            raw_text="\n".join(current_lines),
            file_type="docx",
        ))

    return pages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_document(file_path: str | Path) -> list[PageText]:
    """
    Điểm vào chính: nhận đường dẫn file (PDF hoặc DOCX),
    trả về list[PageText] theo từng trang/section.

    Parameters
    ----------
    file_path : Đường dẫn tới file PDF hoặc DOCX.

    Returns
    -------
    list[PageText] — danh sách văn bản thô theo trang.

    Raises
    ------
    ValueError       : Định dạng file không hỗ trợ.
    FileNotFoundError: File không tồn tại.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tìm thấy: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _parse_docx(path)
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {suffix}. Chỉ hỗ trợ .pdf và .docx")

