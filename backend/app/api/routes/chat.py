"""PoC: exactly one supported AVL sample question, answered using fixed mock data."""
from fastapi import APIRouter, HTTPException, status

from app.schemas.contracts import ChatRequest, Citation, RAGResponse

router = APIRouter()
SAMPLE_PROMPT = "Cây AVL là gì và độ phức tạp tìm kiếm là bao nhiêu?"


@router.post("/api/chat", response_model=RAGResponse, tags=["Chat Agent"])
async def chat_mock_endpoint(request: ChatRequest):
    question = " ".join(request.query.strip().split())
    if not question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    if question != SAMPLE_PROMPT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PoC hiện chỉ hỗ trợ câu hỏi AVL mẫu được hiển thị trên giao diện.",
        )

    return RAGResponse(
        query=question,
        answer=(
            "[MOCK - PoC] Cây AVL là cây nhị phân tìm kiếm (BST) tự cân bằng.\n\n"
            "- Hệ số cân bằng của mỗi nút thuộc {-1, 0, 1}.\n"
            "- Thời gian tìm kiếm trên cây AVL là **O(log n)**."
        ),
        citations=[
            Citation(
                filename="Chuong4_Trees_DSA.pdf",
                page=12,
                section="4.2 Cây AVL - Định nghĩa",
                snippet="[MOCK] Cây AVL là cây nhị phân tìm kiếm tự cân bằng.",
            ),
            Citation(
                filename="Chuong4_Trees_DSA.pdf",
                page=15,
                section="4.2 Cây AVL - Độ phức tạp",
                snippet="[MOCK] Tìm kiếm trên cây AVL có độ phức tạp O(log n).",
            ),
        ],
        confidence_score=0.95,  # Fixed demo value, not computed
        is_safe=True,            # Fixed demo value, no guardrail is running
    )