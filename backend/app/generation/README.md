# Sinh nội dung C2

Ba endpoint POST `/api/chat`, `/api/quiz`, `/api/revision` dùng trường `query`
và nhận `workspace_id`, `corpus_id` tùy chọn. Mặc định là
`default-workspace` và `user-uploads`. Quiz nhận thêm `num_questions` (1–20).
Khi reranker lỗi, RRF được gọi trong cùng scope.

Đặt `LLM_PROVIDER`, `LLM_MODEL` và API key tương ứng trong `backend/.env`.
Để dùng gateway tương thích OpenAI, đặt `LLM_PROVIDER=openai` và
`LLM_BASE_URL` là URL của gateway, chẳng hạn `https://gateway.example/v1`.
Khi thiếu key hoặc provider không trả kết quả, API có thể trả
`generation_mode=extractive`. Quiz trích xuất tạo bài điền từ từ đoạn nguồn,
nên tài liệu ngắn có thể cho ít câu hơn số yêu cầu.

`confidence_score` là điểm hỗ trợ từ nguồn tính theo độ phủ trích dẫn và
relevance trên thang của phương thức retrieval, có phạt câu trả lời mơ hồ.
Điểm này chưa được hiệu chuẩn thành xác suất đúng. Trích dẫn giữ nguyên nhãn
nguồn để giao diện có thể hiển thị [2] dù [1] không được viện dẫn.
