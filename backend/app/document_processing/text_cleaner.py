"""B1: document text cleaning — làm sạch văn bản trích xuất từ PDF/DOCX."""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Các pattern cần loại bỏ
# ---------------------------------------------------------------------------

# Header/Footer phổ biến trong tài liệu DSA tiếng Việt và tiếng Anh
_HEADER_FOOTER_PATTERNS: list[re.Pattern[str]] = [
    # "Trang 5", "Page 5 of 20", "- 5 -"
    re.compile(r"(?m)^(Trang|Page)\s*\d+(\s*(of|/)\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"(?m)^-\s*\d+\s*-$"),
    # Chương X (chỉ dòng riêng, không có nội dung)
    re.compile(r"(?m)^(Chương|Chapter)\s*\d+\s*$", re.IGNORECASE),
    # Dấu bản quyền cuối trang
    re.compile(r"(?m)^©.{0,120}$"),
    # Dòng chỉ chứa số
    re.compile(r"(?m)^\d+\s*$"),
]

# Ký tự Unicode thay thế / bullet fonts lạ → chuẩn hóa
_UNICODE_BULLET_RE = re.compile(r"[\uf0b7\uf0a7\u2022\u25cf\u25aa\u2013\u2014]")
# Ký tự điều khiển (giữ \n \t)
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Hyphenation cuối dòng: "struc-\nture" → "structure"
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
# Whitespace ngang thừa (space + tab), không xóa newline
_HSPACE_RE = re.compile(r"[ \t]+")
# 3+ newlines liên tiếp → 2 newlines (giữ ranh giới đoạn)
_MULTI_NL_RE = re.compile(r"\n{3,}")

# Code block guard: indented block (≥4 spaces) hoặc monospace fence
# Dùng để **tránh** làm sạch bên trong code
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INDENTED_BLOCK_RE = re.compile(r"(?m)^( {4}|\t).+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(raw_text: str, language: str = "unknown") -> str:
    """
    Làm sạch văn bản thô từ PDF/DOCX.

    Thứ tự xử lý:
      1. Chuẩn hóa Unicode (NFC) — quan trọng với tiếng Việt
      2. Xóa ký tự điều khiển
      3. Nối từ bị ngắt dòng kiểu hyphen
      4. Thay thế bullet font lạ bằng dấu gạch đầu dòng "-"
      5. Xóa header/footer lặp lại
      6. Rút gọn khoảng trắng ngang thừa
      7. Rút gọn dòng trống thừa

    Parameters
    ----------
    raw_text : Văn bản thô từ document_parser.
    language : Mã ngôn ngữ ("vi", "en", "unknown") — dự phòng cho xử lý đặc thù.

    Returns
    -------
    Chuỗi đã làm sạch.
    """
    if not raw_text or not raw_text.strip():
        return ""

    # --- Bước 1: Chuẩn hóa Unicode NFC (quan trọng cho tiếng Việt tổ hợp dấu) ---
    text = unicodedata.normalize("NFC", raw_text)

    # --- Bước 2: Xóa ký tự điều khiển ---
    text = _CTRL_CHARS_RE.sub("", text)

    # --- Bước 3: Nối từ bị hyphen xuống dòng ---
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    # --- Bước 4: Chuẩn hóa bullet lạ ---
    text = _UNICODE_BULLET_RE.sub("-", text)

    # --- Bước 5: Xóa header/footer ---
    for pattern in _HEADER_FOOTER_PATTERNS:
        text = pattern.sub("", text)

    # --- Bước 6: Rút gọn khoảng trắng ngang ---
    # (không làm trong code block để giữ indentation)
    text = _HSPACE_RE.sub(" ", text)

    # --- Bước 7: Rút gọn dòng trống thừa ---
    text = _MULTI_NL_RE.sub("\n\n", text)

    return text.strip()


def is_valid_chunk_text(text: str, min_words: int = 20) -> bool:
    """
    Kiểm tra xem text có đáng được index không.

    - Phải có ít nhất `min_words` từ.
    - Không được là toàn số / ký tự đặc biệt.
    - Không rỗng sau khi strip.
    """
    if not text or not text.strip():
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    # Nếu >80% token là số/ký tự không phải chữ → nhiễu
    non_alpha = sum(1 for w in words if not re.search(r"[a-zA-ZÀ-ỹ]", w))
    if non_alpha / len(words) > 0.8:
        return False
    return True

