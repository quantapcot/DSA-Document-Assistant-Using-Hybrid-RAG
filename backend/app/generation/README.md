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
  từng câu, kiểm tra lựa chọn và cung cấp đoạn trích có thể đối chiếu. API
  trả `requested_count` và `generated_count`. Streamlit hiển thị giải thích
  và đáp án sau khi nộp. API phục vụ chấm bài nên vẫn trả đáp án cho client.
- Revision nhận `exam_date` hoặc `days_remaining` (1–90), `minutes_per_day`
  (15–240), `level` và `query` hoặc `all_topics=true`. Lịch ưu tiên các
  chương theo thứ tự tài liệu và dùng tối đa 10 mục mỗi lần để giới hạn
  đầu ra. Các mục chưa xếp lịch được báo riêng. Checklist chỉ được lưu
  trong phiên Streamlit.
- Nếu thiếu FIT-AI hoặc không có đầu ra đạt kiểm tra, Quiz/Revision trả
  `generation_mode=unavailable` cùng thông báo. Workspace rỗng trả
  `generation_mode=abstained`.

`confidence_score` là điểm hỗ trợ từ nguồn tính theo độ phủ trích dẫn và
relevance trên thang của phương thức retrieval, có phạt câu trả lời mơ hồ.
Điểm này chưa được hiệu chuẩn thành xác suất đúng. Trích dẫn giữ nguyên nhãn
nguồn để giao diện có thể hiển thị [2] dù [1] không được viện dẫn.
