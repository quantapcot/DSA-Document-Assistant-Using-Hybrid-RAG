# dsa_ai_agent/backend/app/api/routes/chat.py
from fastapi import APIRouter, HTTPException, status
from app.schemas.contracts import ChatRequest, RAGResponse, Citation

router = APIRouter()

@router.post("/api/chat", response_model=RAGResponse, tags=["Chat Agent"])
async def chat_mock_endpoint(request: ChatRequest):
    """
    Endpoint xử lý trò chuyện hỏi đáp DSA giả lập (Mock API cho Tuần 4).
    Giúp Frontend (A2) kết nối giao diện và test khung Trích dẫn (Citations).
    """
    user_query = request.query.strip()
    
    if not user_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Câu hỏi không được để trống."
        )
        
    # Tạo dữ liệu giả lập (Mock Data) chuẩn theo Pydantic Contract RAGResponse
    mock_citations = [
        Citation(
            filename="Chuong4_Trees_DSA.pdf",
            page=12,
            section="4.2 Cây AVL - Định nghĩa",
            snippet="Cây AVL là cây nhị phân tìm kiếm tự cân bằng, trong đó độ cao của hai cây con không lệch quá 1."
        ),
        Citation(
            filename="Chuong4_Trees_DSA.pdf",
            page=15,
            section="4.2 Cây AVL - Độ phức tạp",
            snippet="Độ phức tạp thời gian cho các thao tác Tìm kiếm, Thêm, Xóa trên Cây AVL đều đạt O(log n)."
        )
    ]
    
    return RAGResponse(
        query=user_query,
        answer=(
            f"[MOCK RESPONSE - TUẦN 4] Dưới đây là giải đáp cho câu hỏi: '{user_query}'\n\n"
            "1. **Khái niệm:** Cây AVL là cây nhị phân tìm kiếm (BST) tự cân bằng.\n"
            "2. **Đặc điểm:** Hệ số cân bằng của mọi nút chỉ nhận các giá trị -1, 0, hoặc 1.\n"
            "3. **Độ phức tạp:** Thao tác tìm kiếm đạt hiệu năng O(log n)."
        ),
        citations=mock_citations,
        confidence_score=0.95,
        is_safe=True
    )