# dsa_ai_agent/backend/app/schemas/contracts.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ==========================================
# 1. CONTRACT B -> C (Team B bàn giao cho Team C)
# ==========================================
class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Mã hash hoặc UUID duy nhất của đoạn văn")
    document_id: str = Field(..., description="ID tài liệu DSA (vd: 'slide_chuong_4_tree')")
    filename: str = Field(..., description="Tên file PDF gốc")
    page: int = Field(..., description="Số trang tương ứng trong slide")
    section: Optional[str] = Field(None, description="Tên tiêu đề mục (vd: '4.2 Cây AVL')")
    text: str = Field(..., description="Nội dung văn bản đã làm sạch")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata bổ sung")

# ==========================================
# 2. CONTRACT C -> A (Team C trả về cho Team A qua API)
# ==========================================
class Citation(BaseModel):
    filename: str = Field(..., description="Tên file slide DSA làm căn cứ")
    page: int = Field(..., description="Số trang trong slide")
    section: Optional[str] = Field(None, description="Tên tiêu đề mục trong slide")
    snippet: str = Field(..., description="Trích đoạn văn bản dùng làm bằng chứng")

class RAGResponse(BaseModel):
    query: str = Field(..., description="Câu hỏi đầu vào của sinh viên")
    answer: str = Field(..., description="Câu trả lời tổng hợp từ RAG Agent")
    citations: List[Citation] = Field(default_factory=list, description="Danh sách trích dẫn nguồn gốc")
    confidence_score: float = Field(..., description="Độ tin cậy của câu trả lời (0.0 đến 1.0)")
    is_safe: bool = Field(True, description="Cờ đánh giá an toàn từ Guardrail")

# Schema cho Request đầu vào của API Chat
class ChatRequest(BaseModel):
    query: str = Field(
        ..., 
        description="Câu hỏi DSA của sinh viên", 
        json_schema_extra={"example": "Cây AVL là gì và độ phức tạp tìm kiếm là bao nhiêu?"}
    )

# ==========================================
# 3. CONTRACT D -> A/C (Team D gán nhãn Guardrail & Safety)
# ==========================================
class GuardrailResult(BaseModel):
    is_safe: bool = Field(..., description="Trạng thái an toàn của câu hỏi")
    risk_category: Optional[str] = Field(None, description="Loại rủi ro: 'prompt_injection', 'unsafe_content'")
    refusal_message: Optional[str] = Field(None, description="Thông điệp từ chối nếu không an toàn")