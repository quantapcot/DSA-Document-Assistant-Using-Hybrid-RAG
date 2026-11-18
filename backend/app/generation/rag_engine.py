"""C2: RAG Engine — điểm vào duy nhất tổng hợp evidence → LLM → answer.

Luồng:
  evidence chunks  →  abstention check  →  format context  →  LLM generate  →  answer str

Nếu LLM không được cấu hình (không có API key), engine tự động fallback
sang chế độ Extractive (ghép text thô từ Top-3 chunks), đảm bảo hệ thống
luôn trả về kết quả dù chưa có LLM.
"""
from __future__ import annotations

import logging

from .abstention import should_abstain, ABSTENTION_RESPONSE
from .citations import format_evidence_for_prompt, evidences_to_citations
from .llm_client import generate_answer
from .prompts.dsa_prompts import get_system_prompt, build_user_prompt
from app.schemas.contracts import Citation

logger = logging.getLogger(__name__)


def build_answer(
    query: str,
    evidences: list[dict],
    intent: str = "qa",
    *,
    method_used: str = "hybrid_rerank",
) -> tuple[str, list[Citation], float]:
    """Tổng hợp câu trả lời cuối cùng từ evidence đã truy xuất.

    Parameters
    ----------
    query       : Câu hỏi gốc của sinh viên.
    evidences   : Danh sách chunks trả về từ RetrievalEngine.
    intent      : Loại ý định: ``"qa"`` / ``"quiz"`` / ``"revision"``.
    method_used : Phương thức retrieval đã dùng (để ghi vào câu trả lời).

    Returns
    -------
    answer          : Chuỗi câu trả lời Markdown hoàn chỉnh.
    citations       : Danh sách Citation objects để Frontend hiển thị.
    confidence_score: Điểm tin cậy (0.0 → 1.0).
    """
    # ── 1. Abstention check ─────────────────────────────────────────────────
    if should_abstain(evidences):
        logger.info("Abstaining: không đủ evidence cho query '%s'", query[:60])
        return ABSTENTION_RESPONSE, [], 0.0

    # ── 2. Build citations ───────────────────────────────────────────────────
    citations = evidences_to_citations(evidences)

    # ── 3. Format context cho LLM prompt ────────────────────────────────────
    context_text = format_evidence_for_prompt(evidences, max_chunks=5)
    system_prompt = get_system_prompt(intent)
    user_prompt = build_user_prompt(query, context_text)

    # ── 4. Gọi LLM ──────────────────────────────────────────────────────────
    llm_answer = generate_answer(system_prompt, user_prompt)

    # ── 5. Fallback sang Extractive nếu LLM không khả dụng ──────────────────
    if not llm_answer:
        logger.info(
            "LLM không khả dụng (thiếu API key hoặc lỗi). "
            "Fallback sang Extractive synthesis."
        )
        answer = _extractive_synthesis(evidences, query, method_used, intent)
        confidence = 0.70
    else:
        answer = llm_answer
        confidence = 0.92

    return answer, citations, confidence


# ---------------------------------------------------------------------------
# Extractive fallback (không cần LLM)
# ---------------------------------------------------------------------------

def _extractive_synthesis(
    evidences: list[dict],
    query: str,
    method_used: str,
    intent: str,
) -> str:
    """Ghép text thô Top-3 chunks thành câu trả lời Markdown có trích dẫn."""
    top_snippets: list[str] = []
    for i, ev in enumerate(evidences[:3], 1):
        sec_info = f" — {ev['section']}" if ev.get("section") else ""
        top_snippets.append(
            f"**[{i}] {ev.get('filename', 'N/A')} "
            f"(Trang {ev.get('page', '?')}{sec_info})**\n"
            f"> {ev.get('text', '').strip()}"
        )

    intent_label = {"quiz": "SINH TRẮC NGHIỆM", "revision": "LỘ TRÌNH ÔN TẬP"}.get(
        intent, "HỎI ĐÁP DSA"
    )

    return (
        f"### 📚 Kết quả truy xuất tài liệu — {intent_label}\n\n"
        f"*(Phương thức: `{method_used}` · {len(evidences)} đoạn phù hợp · "
        f"Chế độ: Extractive — chưa có LLM API key)*\n\n"
        + "\n\n".join(top_snippets)
        + "\n\n---\n"
        "💡 **Để nhận câu trả lời được diễn giải tự nhiên bởi AI,** "
        "hãy cấu hình `GEMINI_API_KEY` hoặc `OPENAI_API_KEY` trong file `backend/.env`."
    )
