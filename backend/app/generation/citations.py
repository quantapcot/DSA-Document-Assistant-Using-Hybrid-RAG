"""C2: citations — đóng gói evidence thành Citation object và snippet cho prompt."""
from __future__ import annotations

from app.schemas.contracts import Citation


SNIPPET_MAX_CHARS = 250


def evidences_to_citations(evidences: list[dict]) -> list[Citation]:
    """Chuyển danh sách evidence chunks thành Citation objects chuẩn hợp đồng."""
    citations: list[Citation] = []
    for ev in evidences:
        raw_text = ev.get("text", "").strip()
        snippet = raw_text[:SNIPPET_MAX_CHARS]
        if len(raw_text) > SNIPPET_MAX_CHARS:
            snippet += "..."
        citations.append(
            Citation(
                filename=ev.get("filename", "unknown.pdf"),
                page=ev.get("page", 1),
                section=ev.get("section") or "Nội dung tài liệu DSA",
                snippet=snippet,
            )
        )
    return citations


def format_evidence_for_prompt(evidences: list[dict], max_chunks: int = 5) -> str:
    """Định dạng danh sách evidence thành khối văn bản để inject vào System Prompt.

    Mỗi đoạn được đánh số và gán nguồn gốc rõ ràng (file, trang, mục)
    để LLM có thể tham chiếu khi sinh câu trả lời.
    """
    lines: list[str] = []
    for i, ev in enumerate(evidences[:max_chunks], 1):
        sec = ev.get("section") or "Nội dung"
        ref = f"[{i}] {ev.get('filename', 'N/A')} — Trang {ev.get('page', '?')} — {sec}"
        lines.append(f"{ref}\n{ev.get('text', '').strip()}")
    return "\n\n---\n\n".join(lines)
