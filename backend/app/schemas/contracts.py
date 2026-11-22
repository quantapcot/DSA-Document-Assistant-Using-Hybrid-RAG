# dsa_ai_agent/backend/app/schemas/contracts.py
from datetime import date
from typing import Annotated, List, Optional, Dict, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    label: int | None = None

class RAGResponse(BaseModel):
    query: str = Field(..., description="Câu hỏi đầu vào của sinh viên")
    answer: str = Field(..., description="Câu trả lời tổng hợp từ RAG Agent")
    citations: List[Citation] = Field(default_factory=list, description="Danh sách trích dẫn nguồn gốc")
    confidence_score: float = Field(..., description="Độ tin cậy của câu trả lời (0.0 đến 1.0)")
    is_safe: bool = Field(True, description="Cờ đánh giá an toàn từ Guardrail")
    generation_mode: str | None = None
    generation_error: str | None = None

# Schema cho Request đầu vào của API Chat
class ChatRequest(BaseModel):
    query: str = Field(
        ..., 
        description="Câu hỏi DSA của sinh viên", 
        json_schema_extra={"example": "Cây AVL là gì và độ phức tạp tìm kiếm là bao nhiêu?"}
    )
    workspace_id: Optional[str] = Field(None, description="Mã workspace của người dùng (tùy chọn)")
    corpus_id: Optional[str] = Field(None, description="Mã corpus cần tìm kiếm (tùy chọn)")


Difficulty = Literal["Easy", "Medium", "Hard"]
QuestionType = Literal["Concept", "Complexity", "Algorithm Tracing", "Comparison", "Application"]
StudyLevel = Literal["Beginner", "Intermediate", "Advanced"]
SourceLabel = Annotated[int, Field(strict=True, ge=1)]


class QuizRequest(ChatRequest):
    num_questions: int = Field(5, ge=1, le=5)
    difficulty: Difficulty = "Medium"


class QuizQuestion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3, strict=True)
    explanation: str = Field(min_length=1)
    sources: list[SourceLabel] = Field(min_length=1, max_length=3)
    question_type: QuestionType
    difficulty: Difficulty
    topic: str = Field(min_length=1, max_length=150)


class QuizResponse(RAGResponse):
    questions: list[QuizQuestion] = Field(default_factory=list)
    requested_count: int = 0
    generated_count: int = 0


class RevisionRequest(ChatRequest):
    query: str = Field("", max_length=500)
    exam_date: date | None = None
    days_remaining: int | None = Field(None, ge=1, le=90)
    minutes_per_day: int = Field(45, ge=15, le=240)
    level: StudyLevel = "Intermediate"
    all_topics: bool = False

    @model_validator(mode="after")
    def validate_planning_input(self):
        if (self.exam_date is None) == (self.days_remaining is None):
            raise ValueError("Provide an exam date or remaining days, not both")
        if not self.all_topics and not self.query.strip():
            raise ValueError("Provide a topic or select all topics")
        return self


class StudyTask(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    kind: Literal["learn", "review", "mock_test"]
    topic_id: str = Field(min_length=1)
    covered_topic_ids: list[str] = Field(default_factory=list)
    objective: str = Field(min_length=1, max_length=300)
    knowledge: list[str] = Field(min_length=1, max_length=5)
    activity: str = Field(min_length=1, max_length=400)
    practice: str = Field(min_length=1, max_length=300)
    minutes: int = Field(ge=5, le=240, strict=True)
    prerequisites: list[str] = Field(default_factory=list)
    sources: list[int] = Field(min_length=1, max_length=4)
    support_quote: str = Field(min_length=20, max_length=500)


class StudyDay(BaseModel):
    date: date
    tasks: list[StudyTask] = Field(min_length=1)


class StudyTopic(BaseModel):
    id: str
    title: str
    sources: list[int]


class StudyPlan(BaseModel):
    exam_date: date
    minutes_per_day: int = Field(ge=15, le=240)
    level: StudyLevel
    days: list[StudyDay] = Field(min_length=1)
    topics: list[StudyTopic]


class RevisionResponse(RAGResponse):
    plan: StudyPlan | None = None
    uncovered_topics: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

# ==========================================
# 3. CONTRACT D -> A/C (Team D gán nhãn Guardrail & Safety)
# ==========================================
class GuardrailResult(BaseModel):
    is_safe: bool = Field(..., description="Trạng thái an toàn của câu hỏi")
    risk_category: Optional[str] = Field(None, description="Loại rủi ro: 'prompt_injection', 'unsafe_content'")
    refusal_message: Optional[str] = Field(None, description="Thông điệp từ chối nếu không an toàn")
