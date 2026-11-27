"""B1: Structure-aware Adaptive Chunking — chia đoạn văn bản theo cấu trúc nội dung và loại tài liệu DSA.

Chiến lược: Structure-aware Adaptive Chunking với token-aware upper bound
---------------------------------------------------------------------------
Thay vì cắt cứng theo số ký tự cố định (512 chars), mỗi đoạn được phân loại
thành một trong các CONTENT_TYPE và áp dụng ngưỡng riêng (chars) phù hợp
với tính chất ngữ nghĩa của nội dung DSA:

  DEFINITION  : Định nghĩa, khái niệm ngắn      → tối đa 400 chars
  ALGORITHM   : Mô tả giải thuật, bước thực hiện → tối đa 600 chars
  CODE        : Pseudocode, code block C++/Python → tối đa 800 chars (giữ nguyên)
  TABLE       : Bảng độ phức tạp, so sánh        → tối đa 500 chars (giữ nguyên)
  EXPLANATION : Giải thích lý thuyết dài          → tối đa 700 chars
  DEFAULT     : Mặc định / không xác định        → tối đa 512 chars

Overlap giữa các chunk:  80 chars (nhỏ hơn 100 để giảm nhiễu, vẫn đủ context)
Token upper bound:        128 tokens — phù hợp với paraphrase-multilingual-MiniLM-L12-v2
                          (mỗi char tiếng Anh ≈ 0.25 token; tiếng Việt ≈ 0.5 token)

Quy trình xử lý:
  1. Phát hiện loại nội dung (content type detection)
  2. Phân đoạn theo ranh giới cấu trúc (heading / paragraph / sentence)
  3. Áp dụng ngưỡng kích thước theo content type
  4. Thêm overlap giữa các chunk liền kề
  5. Lọc chunk rác (< MIN_CHUNK_CHARS)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Content Type Enum & Token / Char limits
# ---------------------------------------------------------------------------

class ContentType(str, Enum):
    DEFINITION  = "definition"   # Định nghĩa, khái niệm
    ALGORITHM   = "algorithm"    # Mô tả giải thuật
    CODE        = "code"         # Code / pseudocode block
    TABLE       = "table"        # Bảng
    EXPLANATION = "explanation"  # Giải thích lý thuyết
    DEFAULT     = "default"      # Mặc định


# Chunk size (characters) theo từng content type
_CHUNK_SIZE: dict[ContentType, int] = {
    ContentType.DEFINITION:  400,
    ContentType.ALGORITHM:   600,
    ContentType.CODE:        800,
    ContentType.TABLE:       500,
    ContentType.EXPLANATION: 700,
    ContentType.DEFAULT:     512,
}

# Overlap (characters) — dùng chung cho mọi loại
CHUNK_OVERLAP: int = 80

# Chunk tối thiểu (nhỏ hơn → bỏ qua, coi là noise)
MIN_CHUNK_CHARS: int = 80

# Token upper bound của embedding model (paraphrase-multilingual-MiniLM-L12-v2)
_TOKEN_UPPER_BOUND: int = 128

# Hệ số ước lượng chars-per-token (thực tế cần verify bằng tokenizer)
# Tiếng Việt ~0.5 token/char, Tiếng Anh ~0.25 token/char → chọn hệ số trung bình
_CHARS_PER_TOKEN_EST: float = 0.4   # 1 token ≈ 2.5 chars (conservative)
_MAX_CHARS_TOKEN_BOUND: int = int(_TOKEN_UPPER_BOUND / _CHARS_PER_TOKEN_EST)  # ≈ 320 chars

# ---------------------------------------------------------------------------
# Content Type Detection (Regex-based)
# ---------------------------------------------------------------------------

# Code block: fenced (```) hoặc indent ≥ 4 spaces / 1 tab trên nhiều dòng
_CODE_FENCE_RE   = re.compile(r"^```", re.MULTILINE)
_CODE_INDENT_RE  = re.compile(r"(?m)^(?: {4}|\t)\S")
# Pseudocode keywords
_PSEUDOCODE_KW   = re.compile(
    r"\b(procedure|function|algorithm|begin|end|if|then|else|while|for|"
    r"do|return|input|output|swap|push|pop|enqueue|dequeue|insert|delete|"
    r"search|merge|sort|heapify|relax|visit|mark|unmark|initialize)\b",
    re.IGNORECASE,
)

# Table: dòng chứa | hoặc dấu gạch ngang liên tiếp (ASCII table)
_TABLE_RE = re.compile(r"(?m)^.*\|.*\|.*$|^[-=|+]{5,}$")

# Heading: Markdown heading hoặc heading số kiểu "4.2 Cây AVL"
_HEADING_RE = re.compile(
    r"^(#{1,4}\s+.+|(?:\d+\.)+\d*\s+.{3,80})$",
    re.MULTILINE,
)

# Definition markers (tiếng Việt + tiếng Anh)
_DEFINITION_RE = re.compile(
    r"\b(là|được định nghĩa|định nghĩa|khái niệm|is defined as|is called|"
    r"refers to|definition|theorem|lemma|corollary|property)\b",
    re.IGNORECASE,
)

# Complexity / algorithm markers
_ALGORITHM_RE = re.compile(
    r"\b(O\(|Θ\(|Ω\(|bước|step|complexity|time complexity|space complexity|"
    r"thuật toán|giải thuật|algorithm|iteration|recursive|đệ quy|vòng lặp|loop)\b",
    re.IGNORECASE,
)

# Explanation markers (longer theoretical text)
_EXPLANATION_RE = re.compile(
    r"\b(giải thích|lý thuyết|chứng minh|chứng tỏ|ta có|ta thấy|"
    r"tổng quát|nhận xét|ví dụ|example|note that|observe that|proof|"
    r"consider|intuitively|formally)\b",
    re.IGNORECASE,
)


def detect_content_type(text: str) -> ContentType:
    """
    Phân loại một đoạn văn bản thành ContentType dựa trên các tín hiệu cấu trúc.

    Thứ tự ưu tiên: CODE > TABLE > DEFINITION > ALGORITHM > EXPLANATION > DEFAULT
    """
    # Ưu tiên cao nhất: CODE
    if _CODE_FENCE_RE.search(text):
        return ContentType.CODE
    code_indent_lines = _CODE_INDENT_RE.findall(text)
    pseudocode_hits   = _PSEUDOCODE_KW.findall(text)
    if len(code_indent_lines) >= 2 or len(pseudocode_hits) >= 3:
        return ContentType.CODE

    # TABLE
    table_lines = _TABLE_RE.findall(text)
    if len(table_lines) >= 2:
        return ContentType.TABLE

    # DEFINITION
    definition_hits = _DEFINITION_RE.findall(text)
    if len(definition_hits) >= 1 and len(text) < 500:
        return ContentType.DEFINITION

    # ALGORITHM
    algo_hits = _ALGORITHM_RE.findall(text)
    if len(algo_hits) >= 2:
        return ContentType.ALGORITHM

    # EXPLANATION
    explanation_hits = _EXPLANATION_RE.findall(text)
    if len(explanation_hits) >= 1:
        return ContentType.EXPLANATION

    return ContentType.DEFAULT


# ---------------------------------------------------------------------------
# Structural segmentation
# ---------------------------------------------------------------------------

def _split_into_structural_segments(text: str) -> list[str]:
    """
    Tách văn bản thành các đoạn theo ranh giới cấu trúc:
      1. Code fence (``` ... ```) → không tách bên trong
      2. Heading (## / 1.2 Title) → ranh giới mạnh
      3. Paragraph break (\\n\\n)   → ranh giới trung bình
      4. Sentence boundary (. / ? / !)  → ranh giới yếu

    Trả về list các đoạn văn bản thô (chưa áp dụng size limit).
    """
    segments: list[str] = []

    # Tách code fence ra trước để không bị cắt vụn
    code_fence_pattern = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
    parts = code_fence_pattern.split(text)

    for part in parts:
        if part.startswith("```") and part.endswith("```"):
            # Giữ nguyên code block như 1 segment
            if part.strip():
                segments.append(part.strip())
            continue

        # Tách theo heading → ranh giới mạnh
        heading_split = re.split(r"(?m)(?=^(?:#{1,4}\s+|(?:\d+\.)+\d*\s+)\S)", part)
        for hs in heading_split:
            if not hs.strip():
                continue
            # Tách theo paragraph (\\n\\n)
            para_split = re.split(r"\n{2,}", hs)
            for para in para_split:
                para = para.strip()
                if para:
                    segments.append(para)

    return [s for s in segments if s.strip()]


def _effective_chunk_size(content_type: ContentType) -> int:
    """
    Trả về chunk size effective = min(type-specific limit, token-bound limit).
    Đảm bảo chunk không vượt quá token upper bound của embedding model.
    """
    type_limit  = _CHUNK_SIZE[content_type]
    return min(type_limit, _MAX_CHARS_TOKEN_BOUND)


# ---------------------------------------------------------------------------
# Adaptive splitting within a segment
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n", ". ", "? ", "! ", "，", "。", " ", ""]


def _adaptive_split_segment(segment: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Nếu segment vượt quá chunk_size, cắt nhỏ hơn theo separator ưu tiên.
    Thêm overlap prefix từ chunk trước.
    """
    if len(segment) <= chunk_size:
        return [segment]

    chunks: list[str] = []
    current = ""

    for sep in _SEPARATORS:
        if sep not in segment and sep != "":
            continue
        parts = segment.split(sep) if sep else list(segment)
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = part if len(part) <= chunk_size else part[:chunk_size]
        if current:
            chunks.append(current.strip())

        # Nếu đã chia được thành nhiều chunk → dừng tìm separator nhỏ hơn
        if len(chunks) > 1:
            break

    if not chunks:
        # Fallback: hard cut
        for i in range(0, len(segment), chunk_size - overlap):
            part = segment[i: i + chunk_size].strip()
            if part:
                chunks.append(part)

    # Thêm overlap
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        combined = (tail + " " + chunks[i]).strip()
        overlapped.append(combined)
    return overlapped


# ---------------------------------------------------------------------------
# Section heading extraction
# ---------------------------------------------------------------------------

def _extract_section(text: str) -> str | None:
    """Lấy heading đầu tiên trong text làm section label."""
    match = _HEADING_RE.search(text)
    if match:
        return match.group(0).lstrip("#").strip()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveChunk:
    """Kết quả của một chunk từ adaptive chunking."""
    text: str
    section: str | None
    content_type: str    # value của ContentType enum


def chunk_text(
    clean_text: str,
    # Các tham số dưới đây được giữ để backward-compat nhưng KHÔNG dùng nữa.
    # Kích thước được quyết định động theo content type.
    chunk_size: int = 512,       # ignored — kept for API compatibility
    chunk_overlap: int = 80,
) -> list[dict[str, str | None]]:
    """
    Chia một chuỗi văn bản đã làm sạch thành list các chunk theo chiến lược
    Structure-aware Adaptive Chunking.

    Parameters
    ----------
    clean_text   : Văn bản đã qua text_cleaner.clean_text().
    chunk_size   : (Không dùng) Giữ lại để tương thích API cũ.
    chunk_overlap: Số ký tự overlap giữa 2 chunk liền kề. Default 80.

    Returns
    -------
    list[dict] với các key:
        - "text"         : Nội dung chunk.
        - "section"      : Heading gần nhất (nếu tìm được).
        - "content_type" : Loại nội dung phát hiện được.
    """
    if not clean_text or not clean_text.strip():
        return []

    # Bước 1: Tách thành các segment theo cấu trúc
    segments = _split_into_structural_segments(clean_text)

    result: list[dict[str, str | None]] = []
    last_section: str | None = None

    for segment in segments:
        # Bước 2: Phát hiện content type của segment này
        ctype = detect_content_type(segment)
        effective_size = _effective_chunk_size(ctype)

        # Bước 3: Phân tách segment nếu vượt kích thước cho phép
        sub_chunks = _adaptive_split_segment(segment, effective_size, chunk_overlap)

        for chunk_str in sub_chunks:
            chunk_str = chunk_str.strip()

            # Bước 4: Lọc chunk quá ngắn
            if len(chunk_str) < MIN_CHUNK_CHARS:
                continue

            # Bước 5: Cập nhật section từ heading trong chunk
            detected_section = _extract_section(chunk_str)
            if detected_section:
                last_section = detected_section

            result.append({
                "text":         chunk_str,
                "section":      last_section,
                "content_type": ctype.value,
            })

    return result


# ---------------------------------------------------------------------------
# Legacy-compat aliases
# ---------------------------------------------------------------------------

def chunk_text_legacy(
    clean_text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[dict[str, str | None]]:
    """
    Giữ nguyên Recursive Character Chunking (512/100) làm baseline đối chiếu.
    Dùng khi cần chạy ablation so sánh với adaptive chunking.
    """
    if not clean_text or not clean_text.strip():
        return []

    _SEPS = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not text or len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        for sep in seps:
            if sep == "" or sep in text:
                parts = text.split(sep) if sep else list(text)
                chunks: list[str] = []
                cur = ""
                for p in parts:
                    cand = (cur + sep + p) if cur else p
                    if len(cand) <= chunk_size:
                        cur = cand
                    else:
                        if cur:
                            chunks.append(cur.strip())
                        cur = p
                if cur:
                    chunks.append(cur.strip())
                remaining = seps[seps.index(sep) + 1:]
                out: list[str] = []
                for c in chunks:
                    if len(c) > chunk_size and remaining:
                        out.extend(_split(c, remaining))
                    elif c.strip():
                        out.append(c.strip())
                return out
        return [text[i: i + chunk_size].strip()
                for i in range(0, len(text), chunk_size - chunk_overlap)]

    raw = _split(clean_text, _SEPS)
    last_sec: str | None = None
    out = []
    for c in raw:
        if len(c) < MIN_CHUNK_CHARS:
            continue
        s = _extract_section(c)
        if s:
            last_sec = s
        out.append({"text": c, "section": last_sec, "content_type": "default"})
    return out
