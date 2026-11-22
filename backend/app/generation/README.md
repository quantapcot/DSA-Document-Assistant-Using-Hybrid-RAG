# Sinh nội dung C2

Ba endpoint POST `/api/chat`, `/api/quiz`, `/api/revision` nhận
`workspace_id`, `corpus_id` tùy chọn. Mặc định là `default-workspace` và
`user-uploads`. Khi reranker lỗi, RRF được gọi trong cùng scope.

Quiz và Revision gọi FIT-AI qua giao diện tương thích OpenAI. Đặt
`LLM_BASE_URL` bằng URL được cấp, `LLM_MODEL` bằng ID Qwen-27B trên gateway
và `OPENAI_API_KEY` bằng key tương ứng trong `backend/.env`. Chat tiếp tục
dùng `LLM_PROVIDER` và có chế độ trích xuất nếu model không khả dụng.

- Quiz nhận `query`, `num_questions` (1–20) và `difficulty` (Easy, Medium,
  Hard). Model sinh tối đa năm câu mỗi batch. Một lượt gọi độc lập giải lại
  từng câu chỉ từ câu hỏi, lựa chọn và nguồn. Lượt tiếp theo kiểm tra giải thích
  sau khi đã đối chiếu đáp án và trích đoạn. Có một lượt sinh bù cho ô còn thiếu. API
  trả `requested_count` và `generated_count`. Streamlit hiển thị giải thích
  và đáp án sau khi nộp. API phục vụ chấm bài nên vẫn trả đáp án cho client.
- Revision nhận `exam_date` hoặc `days_remaining` (1–90), `minutes_per_day`
  (15–240), `level` và `query` hoặc `all_topics=true`. Bộ lập lịch Python
  ưu tiên theo quan hệ tiên quyết suy ra từ tài liệu và chọn thời lượng theo
  độ phức tạp, trình độ. FIT-AI sinh nội dung theo batch tối đa ba nhiệm vụ.
  Batch lỗi JSON hoặc bị cắt được thử lại theo từng nhiệm vụ với cùng ID
  và nguồn. Các mục chưa xếp lịch được báo riêng. Checklist chỉ lưu trong
  phiên Streamlit.
- Nếu thiếu FIT-AI hoặc không có đầu ra đạt kiểm tra, Quiz/Revision trả
  `generation_mode=unavailable` cùng thông báo. Workspace rỗng trả
  `generation_mode=abstained`.

Các thiết lập vận hành trong `backend/.env`:

| Nhóm | Biến | Mặc định |
| --- | --- | --- |
| Timeout mỗi lần gọi model (giây) | `LLM_CHAT_TIMEOUT`, `LLM_QUIZ_GENERATION_TIMEOUT`, `LLM_QUIZ_VERIFICATION_TIMEOUT`, `LLM_REVISION_TIMEOUT` | 30, 45, 40, 45 |
| Ngân sách mỗi HTTP request (giây) | `CHAT_REQUEST_TIMEOUT`, `QUIZ_REQUEST_TIMEOUT`, `REVISION_REQUEST_TIMEOUT` | 90, 360, 240 |
| Giới hạn phạm vi lập lịch | `REVISION_MAX_TOPICS`, `REVISION_SOURCE_TOKEN_BUDGET` | 20, 12000 |

Streamlit đọc ngân sách HTTP từ `/health`. Output FIT-AI cho Quiz và Revision
được chặn ở 4.000 token mỗi lần gọi. Ngân sách nguồn của Revision dùng ước
lượng bảo thủ theo kích thước UTF-8. Hết hạn mức không được retry. Chat giữ
fallback trích xuất và gửi lý do cho giao diện qua `generation_error`.

`confidence_score` là điểm hỗ trợ từ nguồn tính theo độ phủ trích dẫn và
relevance trên thang của phương thức retrieval, có phạt câu trả lời mơ hồ.
Điểm này chưa được hiệu chuẩn thành xác suất đúng. Trích dẫn giữ nguyên nhãn
nguồn để giao diện có thể hiển thị [2] dù [1] không được viện dẫn.
