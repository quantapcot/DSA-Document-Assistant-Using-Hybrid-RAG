"""B1: document_parser — PDF/DOCX/Image → list[PageText], có hỗ trợ OCR cho PDF scan và ảnh nhúng."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.document_processing.ocr_engine import extract_text_from_image

logger = logging.getLogger("app.document_processing.document_parser")

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
})

_MIN_EMBEDDED_IMAGE_AREA = 8_000
_OCR_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ0-9]{2,}")


# ---------------------------------------------------------------------------
# Data structure trung gian (trước khi tạo DocumentChunk)
# ---------------------------------------------------------------------------

@dataclass
class PageText:
    """Văn bản thô của một trang/đoạn trong tài liệu."""
    page: int          # 1-indexed
    raw_text: str      # chưa làm sạch
    file_type: str     # "pdf" | "docx" | "image"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_meaningful_ocr(text: str, min_chars: int = 15, min_words: int = 3) -> bool:
    cleaned = text.strip()
    if len(cleaned) < min_chars:
        return False
    return len(_OCR_WORD_RE.findall(cleaned)) >= min_words


def _ocr_bytes_cached(image_bytes: bytes, cache: dict[str, str]) -> str:
    digest = hashlib.sha1(image_bytes).hexdigest()
    if digest in cache:
        return cache[digest]
    text = extract_text_from_image(image_bytes)
    cache[digest] = text
    return text


# ---------------------------------------------------------------------------
# PDF Parser (PyMuPDF + OCR)
# ---------------------------------------------------------------------------

def _count_xref_pages(doc: Any) -> dict[int, int]:
    """Đếm số trang mỗi image xref xuất hiện để lọc bỏ logo/watermark lặp lại."""
    counts: dict[int, int] = {}
    for page in doc:
        seen_on_page = {img[0] for img in page.get_images(full=False)}
        for xref in seen_on_page:
            counts[xref] = counts.get(xref, 0) + 1
    return counts


def _extract_pdf_embedded_images_ocr(
    doc: Any,
    page: Any,
    xref_page_counts: dict[int, int],
    image_ocr_cache: dict[str, str],
) -> list[str]:
    """Trích xuất và OCR các ảnh có ý nghĩa nằm trong một trang PDF."""
    num_pages = len(doc)
    repeat_threshold = max(3, int(num_pages * 0.4)) if num_pages >= 5 else num_pages + 1
    min_w = settings.OCR_MIN_IMAGE_WIDTH
    min_h = settings.OCR_MIN_IMAGE_HEIGHT

    seen_xrefs: set[int] = set()
    extracted_texts: list[str] = []

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        width = img_info[2] if len(img_info) > 2 else 0
        height = img_info[3] if len(img_info) > 3 else 0

        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        if width < min_w or height < min_h or (width * height) < _MIN_EMBEDDED_IMAGE_AREA:
            continue
        if xref_page_counts.get(xref, 0) >= repeat_threshold:
            continue

        try:
            extracted = doc.extract_image(xref)
        except Exception as exc:
            logger.debug("Could not extract PDF image xref=%s: %s", xref, exc)
            continue

        img_bytes = extracted.get("image") if extracted else None
        if not img_bytes:
            continue

        ocr_txt = _ocr_bytes_cached(img_bytes, image_ocr_cache)
        if _is_meaningful_ocr(ocr_txt):
            extracted_texts.append(ocr_txt.strip())

    return extracted_texts


def _parse_pdf(
    file_path: Path,
    *,
    use_ocr: bool,
    extract_embedded_images: bool,
    ocr_dpi: int,
    min_page_text_chars: int,
) -> list[PageText]:
    """
    Trích xuất text từ PDF theo từng trang bằng PyMuPDF.

    - Nếu trang là bản scan (ít/không có text chọn được nhưng có ảnh/hình vẽ):
      render toàn trang ở `ocr_dpi` rồi chạy OCR.
    - Nếu trang có text chọn được và bật `extract_embedded_images`:
      OCR thêm các ảnh/biểu đồ/screenshot nhúng trong trang (bỏ qua icon nhỏ & logo lặp).
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError("Cài đặt PyMuPDF: pip install pymupdf") from exc

    pages: list[PageText] = []
    image_ocr_cache: dict[str, str] = {}

    with pymupdf.open(str(file_path)) as doc:
        xref_page_counts = _count_xref_pages(doc) if (use_ocr and extract_embedded_images) else {}
        zoom = ocr_dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)

        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            page_images = page.get_images(full=True) if use_ocr else []
            has_visual_content = bool(page_images) or (use_ocr and bool(page.get_drawings()))

            final_text = text

            if use_ocr and len(text) < min_page_text_chars and has_visual_content:
                # Trang scan hoặc trang dựng từ nhiều mảnh ảnh
                try:
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    ocr_text = extract_text_from_image(pix.tobytes("png")).strip()
                except Exception as exc:
                    logger.warning("Full-page OCR failed on %s page %d: %s", file_path.name, i + 1, exc)
                    ocr_text = ""

                if ocr_text:
                    final_text = ocr_text if len(text) < 10 else f"{text}\n\n{ocr_text}"

            elif use_ocr and extract_embedded_images and page_images:
                # Trang có text bình thường nhưng kèm ảnh nhúng (sơ đồ, code screenshot, bảng ảnh)
                embedded_ocr = _extract_pdf_embedded_images_ocr(
                    doc, page, xref_page_counts, image_ocr_cache
                )
                if embedded_ocr:
                    joined = "\n\n".join(embedded_ocr)
                    final_text = f"{text}\n\n[OCR Image]\n{joined}" if text else joined

            if final_text.strip():
                pages.append(PageText(page=i + 1, raw_text=final_text, file_type="pdf"))

    return pages


# ---------------------------------------------------------------------------
# DOCX Parser (python-docx + OCR ảnh nhúng)
# ---------------------------------------------------------------------------

def _extract_docx_para_images_ocr(
    doc: Any,
    para: Any,
    seen_digests: set[str],
    image_ocr_cache: dict[str, str],
) -> list[str]:
    """Lấy và OCR các ảnh được nhúng bên trong một paragraph của DOCX."""
    r_ids = para._element.xpath(".//*[local-name()='blip']/@*[local-name()='embed']")
    texts: list[str] = []
    for r_id in r_ids:
        part = doc.part.related_parts.get(r_id)
        if part is None:
            continue
        blob = getattr(part, "blob", None)
        if not blob:
            continue
        digest = hashlib.sha1(blob).hexdigest()
        seen_digests.add(digest)
        ocr_txt = _ocr_bytes_cached(blob, image_ocr_cache)
        if _is_meaningful_ocr(ocr_txt):
            texts.append(ocr_txt.strip())
    return texts


def _parse_docx(
    file_path: Path,
    *,
    use_ocr: bool,
    extract_embedded_images: bool,
) -> list[PageText]:
    """
    Trích xuất text từ DOCX theo paragraph, bảng biểu và ảnh nhúng.
    DOCX không có khái niệm trang → tách theo heading H1 để tạo "trang ảo".
    """
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("Cài đặt python-docx: pip install python-docx") from exc

    doc = Document(str(file_path))

    pages: list[PageText] = []
    current_lines: list[str] = []
    current_page = 1
    seen_digests: set[str] = set()
    image_ocr_cache: dict[str, str] = {}

    for para in doc.paragraphs:
        text = para.text.strip()

        # Nếu gặp heading cấp 1 → tạo trang mới
        if text and para.style.name.startswith("Heading 1") and current_lines:
            pages.append(PageText(
                page=current_page,
                raw_text="\n".join(current_lines),
                file_type="docx",
            ))
            current_lines = []
            current_page += 1

        if text:
            current_lines.append(text)

        if use_ocr and extract_embedded_images:
            img_texts = _extract_docx_para_images_ocr(doc, para, seen_digests, image_ocr_cache)
            for ocr_txt in img_texts:
                current_lines.append(f"[OCR Image]\n{ocr_txt}")

    # Bổ sung text từ bảng biểu nếu có
    for table in doc.tables:
        table_rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                table_rows.append(" | ".join(cells))
        if table_rows:
            current_lines.append("\n".join(table_rows))

    # OCR các ảnh còn lại trong DOCX (vd: nằm trong bảng, header hoặc shape)
    if use_ocr and extract_embedded_images:
        for rel in doc.part.rels.values():
            target = getattr(rel, "target_part", None)
            content_type = getattr(target, "content_type", "")
            if not content_type.startswith("image/"):
                continue
            blob = getattr(target, "blob", None)
            if not blob:
                continue
            digest = hashlib.sha1(blob).hexdigest()
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
            ocr_txt = _ocr_bytes_cached(blob, image_ocr_cache)
            if _is_meaningful_ocr(ocr_txt):
                current_lines.append(f"[OCR Image]\n{ocr_txt.strip()}")

    # Flush phần còn lại
    if current_lines:
        pages.append(PageText(
            page=current_page,
            raw_text="\n".join(current_lines),
            file_type="docx",
        ))

    return pages


# ---------------------------------------------------------------------------
# Standalone Image Parser (OCR)
# ---------------------------------------------------------------------------

def _parse_image(file_path: Path, *, use_ocr: bool) -> list[PageText]:
    """Trích xuất văn bản trực tiếp từ file ảnh (.png, .jpg, ...)."""
    if not use_ocr:
        return []
    text = extract_text_from_image(file_path.read_bytes()).strip()
    if not text:
        return []
    return [PageText(page=1, raw_text=text, file_type="image")]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_document(
    file_path: str | Path,
    *,
    use_ocr: bool | None = None,
    extract_embedded_images: bool | None = None,
    ocr_dpi: int | None = None,
    min_page_text_chars: int | None = None,
) -> list[PageText]:
    """
    Điểm vào chính: nhận đường dẫn file (PDF, DOCX hoặc Ảnh),
    trả về list[PageText] theo từng trang/section (có fallback/bổ sung OCR).

    Parameters
    ----------
    file_path               : Đường dẫn tới file PDF, DOCX hoặc file ảnh.
    use_ocr                 : Bật/tắt OCR (mặc định theo `settings.OCR_ENABLED`).
    extract_embedded_images : Bật/tắt OCR ảnh nhúng trên các trang đã có text.
    ocr_dpi                 : Độ phân giải render trang scan PDF.
    min_page_text_chars     : Ngưỡng ký tự tối thiểu trước khi kích hoạt OCR toàn trang.

    Returns
    -------
    list[PageText] — danh sách văn bản thô theo trang.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File không tìm thấy: {file_path}")

    ocr_enabled = settings.OCR_ENABLED if use_ocr is None else use_ocr
    embedded_enabled = (
        settings.OCR_EXTRACT_EMBEDDED_IMAGES
        if extract_embedded_images is None
        else extract_embedded_images
    )
    dpi = settings.OCR_DPI if ocr_dpi is None else ocr_dpi
    min_chars = settings.OCR_MIN_PAGE_TEXT_CHARS if min_page_text_chars is None else min_page_text_chars

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(
            path,
            use_ocr=ocr_enabled,
            extract_embedded_images=embedded_enabled,
            ocr_dpi=dpi,
            min_page_text_chars=min_chars,
        )
    if suffix in (".docx", ".doc"):
        return _parse_docx(
            path,
            use_ocr=ocr_enabled,
            extract_embedded_images=embedded_enabled,
        )
    if suffix in IMAGE_EXTENSIONS:
        return _parse_image(path, use_ocr=ocr_enabled)

    raise ValueError(
        f"Định dạng không hỗ trợ: {suffix}. Chỉ hỗ trợ .pdf, .docx và ảnh ({', '.join(sorted(IMAGE_EXTENSIONS))})"
    )
