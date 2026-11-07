"""B1: structure-aware document chunking — chia đoạn văn bản thành các chunk phù hợp RAG."""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Hằng số mặc định
# ---------------------------------------------------------------------------

# Slide/Handout ngắn: chunk nhỏ hơn để không lẫn nhiều slide vào nhau
DEFAULT_CHUNK_SIZE = 512          # characters (không phải tokens)
DEFAULT_CHUNK_OVERLAP = 100       # characters overlap giữa 2 chunk liền kề
MIN_CHUNK_SIZE = 100              # chunk ngắn hơn thế sẽ bị merge hoặc bỏ

# Separators ưu tiên theo thứ tự: đoạn > dòng > câu > từ
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Regex nhận diện heading Markdown-like hoặc heading số (vd: "4.2 Cây AVL")
_HEADING_RE = re.compile(
    r"^(#{1,4}\s+.+|(?:\d+\.)+\d*\s+.{3,80})$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_by_separator(text: str, sep: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Chia text theo separator, ghép lại thành chunk ≤ chunk_size,
    thêm overlap giữa các chunk liền kề.
    """
    parts = text.split(sep) if sep else list(text)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # Nếu part đơn lẻ đã vượt chunk_size → đệ quy xuống separator nhỏ hơn
            if len(part) > chunk_size:
                # Trả về xử lý ở bên ngoài bằng cách push toàn bộ
                current = part
            else:
                current = part

    if current:
        chunks.append(current.strip())

    # Thêm overlap: mỗi chunk được prefix bởi phần cuối của chunk trước
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        overlapped.append((tail + " " + chunks[i]).strip())
    return overlapped


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Recursive Character Text Splitter — tương tự LangChain nhưng không phụ thuộc package."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    for sep in separators:
        if sep == "" or sep in text:
            chunks = _split_by_separator(text, sep, chunk_size, overlap)
            # Mọi chunk vẫn quá lớn → tiếp tục với separator nhỏ hơn
            remaining_seps = separators[separators.index(sep) + 1:]
            result: list[str] = []
            for chunk in chunks:
                if len(chunk) > chunk_size and remaining_seps:
                    result.extend(_recursive_split(chunk, remaining_seps, chunk_size, overlap))
                elif chunk.strip():
                    result.append(chunk.strip())
            return result

    # Fallback: cắt cứng
    return [text[i : i + chunk_size].strip() for i in range(0, len(text), chunk_size - overlap)]


def _extract_section_from_heading(text: str) -> str | None:
    """Lấy heading đầu tiên trong text làm section label."""
    match = _HEADING_RE.search(text)
    if match:
        return match.group(0).lstrip("#").strip()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    clean_text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, str | None]]:
    """
    Chia một chuỗi văn bản đã làm sạch thành list các chunk.

    Parameters
    ----------
    clean_text   : Văn bản đã qua text_cleaner.clean_text().
    chunk_size   : Kích thước tối đa (characters) của mỗi chunk.
    chunk_overlap: Số ký tự chồng lên nhau giữa 2 chunk liền kề.

    Returns
    -------
    list[dict] với các key:
        - "text"    : Nội dung chunk.
        - "section" : Heading gần nhất (nếu tìm được), dùng làm metadata.
    """
    if not clean_text or not clean_text.strip():
        return []

    raw_chunks = _recursive_split(clean_text, _SEPARATORS, chunk_size, chunk_overlap)

    result: list[dict[str, str | None]] = []
    last_section: str | None = None

    for chunk_text_item in raw_chunks:
        if len(chunk_text_item) < MIN_CHUNK_SIZE:
            continue  # bỏ qua chunk quá ngắn (noise)

        # Thử detect section từ heading bên trong chunk
        detected = _extract_section_from_heading(chunk_text_item)
        if detected:
            last_section = detected

        result.append({"text": chunk_text_item, "section": last_section})

    return result

