"""C2: abstain — phát hiện câu trả lời thiếu bằng chứng và trả lời từ chối rõ ràng.

Hàm ``should_abstain`` được gọi TRƯỚC khi gọi LLM. Nếu bằng chứng yếu hoặc
không có, hệ thống trả về ``abstention_response`` thay vì đẩy context rỗng
vào LLM (tránh hallucination).
"""
from __future__ import annotations


# Ngưỡng điểm tối thiểu để xem chunk là "đủ bằng chứng"
_SCORE_THRESHOLD = 0.25
# Số chunk tối thiểu cần có để không từ chối
_MIN_EVIDENCE_COUNT = 1

ABSTENTION_RESPONSE = (
    "⚠️ **Không tìm thấy nội dung phù hợp trong tài liệu.**\n\n"
    "Hệ thống chưa tìm thấy bằng chứng liên quan đến câu hỏi của bạn "
    "trong kho tài liệu hiện tại.\n\n"
    "💡 **Gợi ý:**\n"
    "- Thử tải lên slide hoặc giáo trình DSA liên quan qua tab **📤 Tải lên tài liệu**.\n"
    "- Điều chỉnh lại từ khóa câu hỏi (ví dụ: thêm tên thuật toán cụ thể).\n"
    "- Kiểm tra xem tài liệu đã được nạp vào hệ thống chưa."
)


def should_abstain(evidences: list[dict]) -> bool:
    """Trả về True nếu hệ thống nên từ chối trả lời vì thiếu bằng chứng.

    Quy tắc:
    1. Không có chunk nào được truy xuất.
    2. Tất cả chunks đều có score dưới ngưỡng ``_SCORE_THRESHOLD``
       (chỉ áp dụng khi score khả dụng — hybrid_rerank gắn score tuyệt đối).
    """
    if not evidences or len(evidences) < _MIN_EVIDENCE_COUNT:
        return True

    # Nếu chunk nào có score (từ reranker), áp dụng kiểm tra ngưỡng
    scored = [ev for ev in evidences if ev.get("score") is not None]
    if scored:
        best_score = max(ev["score"] for ev in scored)
        if best_score < _SCORE_THRESHOLD:
            return True

    return False
