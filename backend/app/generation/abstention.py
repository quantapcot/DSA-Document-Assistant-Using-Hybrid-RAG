"""C2: abstain — phát hiện câu trả lời thiếu bằng chứng và trả lời từ chối rõ ràng.

Hàm ``should_abstain`` được gọi TRƯỚC khi gọi LLM. Nếu bằng chứng yếu hoặc
không có, hệ thống trả về ``abstention_response`` thay vì đẩy context rỗng
vào LLM (tránh hallucination).
"""
from __future__ import annotations
from math import isfinite


# Ngưỡng điểm tối thiểu để xem chunk là "đủ bằng chứng"
SCORE_THRESHOLDS = {"hybrid_rerank": 0.25, "hybrid_rrf": 0.015, "dense": 0.40, "bm25": 5.0}
EMPTY_WORKSPACE_RESPONSE = (
    "Workspace chưa có tài liệu nào. Vui lòng tải tài liệu môn học (PDF/DOCX) để bắt đầu."
)

ABSTENTION_RESPONSE = (
    "⚠️ **Không tìm thấy nội dung phù hợp trong tài liệu.**\n\n"
    "Hệ thống chưa tìm thấy bằng chứng liên quan đến câu hỏi của bạn "
    "trong kho tài liệu hiện tại.\n\n"
    "💡 **Gợi ý:**\n"
    "- Thử tải lên slide hoặc giáo trình DSA liên quan qua tab **📤 Tải lên tài liệu**.\n"
    "- Điều chỉnh lại từ khóa câu hỏi (ví dụ: thêm tên thuật toán cụ thể).\n"
    "- Kiểm tra xem tài liệu đã được nạp vào hệ thống chưa."
)


def evidence_score(evidence: dict) -> float | None:
    score = evidence.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and isfinite(score):
        return float(score)
    return None


def should_abstain(evidences: list[dict], method: str = "hybrid_rerank") -> bool:
    """Apply the cutoff for the retrieval method that supplied the evidence."""
    if not evidences:
        return True
    threshold = SCORE_THRESHOLDS[method]
    scores = [score for evidence in evidences if (score := evidence_score(evidence)) is not None]
    return bool(scores) and max(scores) < threshold
