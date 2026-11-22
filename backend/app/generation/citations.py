"""C2: citations — đóng gói evidence thành Citation object và snippet cho prompt."""
from __future__ import annotations
import re

from app.schemas.contracts import Citation


SNIPPET_MAX_CHARS = 250
MAX_EVIDENCES = 5
_CITATION = re.compile(r"\[([1-9][0-9]{0,8})\]")


def cited_labels(answer: str, evidence_count: int) -> list[int]:
    return list(dict.fromkeys(
        label for match in _CITATION.finditer(answer)
        if (label := int(match[1])) <= evidence_count
    ))


def filter_actual_citations(answer: str, evidences: list[dict]) -> list[Citation]:
    return citations_for_labels(evidences, cited_labels(answer, len(evidences)))


def citations_for_labels(evidences: list[dict], labels: list[int]) -> list[Citation]:
    citations = evidences_to_citations(evidences)
    return [
        citations[label - 1].model_copy(update={"label": label})
        for label in dict.fromkeys(labels) if 1 <= label <= len(citations)
    ]


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


def format_evidence_for_prompt(evidences: list[dict], max_chunks: int = MAX_EVIDENCES) -> str:
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
